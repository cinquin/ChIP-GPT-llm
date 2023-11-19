# Copyright 2023 Olivier Cinquin
# SPDX-License-Identifier: Apache-2.0

# Portions of this file are derived from the following projects
#   - Hugging Face PEFT https://github.com/huggingface/peft
#   - Alpaca-LoRA https://github.com/tloen/alpaca-lora
#   - Stanford Alpaca https://github.com/tatsu-lab/stanford_alpaca
# under Apache License, Version 2.0

import bisect
import copy
import datetime
import gc
import json
import logging
import os
import re
import traceback
import warnings
from collections import namedtuple
from typing import (Dict, Any, Union, TextIO, Sequence, Match)

import numpy as np
import peft
import torch
import transformers
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    PeftModel,
)
from torch import Tensor
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, Trainer
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TrainerControl, TrainerState, TrainingArguments, BatchEncoding
from transformers.integrations import TrainerCallback
from transformers.tokenization_utils import PreTrainedTokenizer

from util import *

logging.basicConfig(level=1)
logger: logging.Logger = logging.getLogger("main_logger")
logger.setLevel(logging.ERROR)

hf_local_files_only: bool = False


class Task(namedtuple("Question", ["id", "dependent_ids", "title", "details", "n_generation_tries", "max_new_tokens", "previous_id", "output_is_input_quote", "forbid_number_prefix", "n_generations"])):
    def __new__(cls, id: int, dependent_ids: List[int], title: str, details: str, n_generation_tries: int = 1,
                max_new_tokens: int = 100, previous_id: int = -1, output_is_input_quote: bool = False,
                forbid_number_prefix: bool = False,
                n_generations: int = 3):
        return super().__new__(cls, id, dependent_ids, title, details, n_generation_tries, max_new_tokens, previous_id, output_is_input_quote, forbid_number_prefix, n_generations)
    
    @property
    def id(self) -> int:
        return self._asdict()['id']
    
    @property
    def previous_id(self) -> int:
        return self._asdict()['previous_id']
    
    @property
    def output_is_input_quote(self) -> int:
        return self._asdict()['output_is_input_quote']
    
    @property
    def dependent_ids(self) -> List[int]:
        return self._asdict()['dependent_ids']
    
    @property
    def title(self) -> str:
        return self._asdict()['title']
    
    @property
    def details(self) -> str:
        return self._asdict()['details']
    
    @property
    def n_generation_tries(self) -> int:
        return self._asdict()['n_generation_tries']
    
    @property
    def max_new_tokens(self) -> int:
        return self._asdict()['max_new_tokens']
    
    @property
    def forbid_number_prefix(self) -> bool:
        return self._asdict()["forbid_number_prefix"]
    
    @property
    def n_generations(self) -> bool:
        return self._asdict()["n_generations"]
    
    @memoize_instance_method
    def details_n_tokens(self, tokenizer):
        return tokenized_length(self.details, tokenizer)


task_data_v2_str: str = \
"""
1\tFalse\t1\t\tCell line name\tName of the cell line used **for this particular sample**, preferably as a single word.  Barb does not include particular modifications introduced in this study. She outputs "primary" if tissue was used instead of an established [immortalized] cell line, or "N/A" if no reference to a specific cell line is provided
2\tFalse\t2\t1\tCell type\tCell type (e.g. fibroblast, cardiomyoblast, monocyte, adenocarcinoma, etc.), as noted in the record or as inferred by Barb from the cell line name. Barb checks for any typos (e.g. "epitheilal" instead of "epithelial") and corrects them
3\tFalse\t3\t1,2\tOrgan\tOrgan of origin denoted in the record or inferred by Barb from the cell line name, preferably as a single word, using the most common term (e.g., lung, PBMC, liver, cornea, ovary, breast)
4\tFalse\t4\t1,2,3\tIntra-organ location\tMore detailed location within the organ (e.g. right atrium auricular region, bronchus, etc.)
5\tFalse\t5\t\tGenetic modifications\tGenetic modifications (e.g. gene knockout, shRNA or RNAi knockdown or silencing, etc.) introduced by the experimenters **for this particular sample**, with names of genes targeted (if any), and excluding wild-type ("WT") genes
6\tFalse\t6\t\tInput control\tDoes the string "input" appear anywhere in the sample name? Is the sample an input control?
7\tFalse\t7\t1,5\tCell name or abbreviation appears in sample name\tDoes the full name of the cells used, or an abbreviation of that name, appear in the sample name?

8\tTrue\t-1\t\tAntibody catalog numbers and manufacturer strings\tQuote any catalog numbers, lot numbers, and manufacturers exactly as they appear in the record (e.g. "Santa Cruz, C-20, sc-1008, lot# H1216")
9\tFalse\t-1\t\tAntibody catalog references\tAntibody catalog references in record, formatted as e.g. manufacturer=santa_cruz,clone=C-20,catalog=sc-1008,lot=H1216,target=VDR
10\tFalse\t-1\t\tHuman gene names or protein complexes mentioned in record\tQuote any human gene names, or human protein complexes, exactly as they appear in the record. If the same gene is mentioned in different ways, choose the form corresponding to the standardized symbol (e.g., prefer "AR" over "Androgen receptor", or "ESR1" over "ER-alpha").

11\tFalse\t8\t1,5,6,7\tBarb's rationale for ChIP target extraction\tBarb's rationale for ChIP target extraction **for this particular sample** from the record and from Barb's own understanding, or for identification as an "input" / empty-vector (not expressing tagged protein) sample. Barb includes the strategy for protein tagging, if relevant, but ignores genetic modifications (e.g. Cas9 gene editing) or genetic background or genetic modifications that do not involve protein tagging of ChIP targets. She thinks step by step, pays particular attention to the sample name, and repeats record entries providing the information as well as words present in the sample name that refer to the ChIP target or "input" and not to the genetic background.
12\tFalse\t9\t6,7,11\tChIP target\tName of ChIP target **for this particular sample**, or "input" if this is an "input" control sample (as indicated, e.g., by the sample name), or if the targeted tag was not actually expressed (e.g., empty vector)
13\tFalse\t10\t12\tHGNC official gene name for ChIP target\tHGNC official human gene name for ChIP target, or "Unsure" if the official name does not appear consistent with the context of the experiment
14\tFalse\t11\t11,12\tSample is generic ChIP-seq\tDoes this sample correspond to generic ChIP-seq? (Barb answers as: ChIP-seq for sure / No, it may be [ATAC-seq, RNA-seq, etc.] / Unsure.)
15\tFalse\t12\t5,11,12\tBarb's rationale for notable treatment extraction\tBarb's rationale for identification of notable treatments applied **to this particular sample** OTHER THAN any genetic modifications (knockout, knockdown, silencing, etc.) already reported above by Barb and OTHER THAN those related to crosslinking, library preparation and sequencing, regular cell culture, etc. Barb includes references to the record entries providing the information, and to relevant words present in the sample name, including possibly "control" if that refers to a *treatment* control instead of a *ChIP input* control; if applicable, Barb compares the sample name to the names of the other samples in the study to identify abbreviations showing which samples had the treatment applied and which did not
16\tFalse\t13\t5,15\tNotable treatments\tNotable treatments applied to **this particular sample** OTHER THAN genetic modifications (knockout, knockdown, silencing, etc.) already reported above, and OTHER THAN those related to crosslinking, library preparation and sequencing, regular cell culture, etc., and formatted as e.g. "cisplatin (concentration=2_uM, duration=3_days, details=DNA_alkylating_agent)". Barb does not report treatments that don't seem to make sense.
17\tFalse\t14\t5,6,15,16\tThis sample received a control genetic modification or has a control genetic background\tDoes this sample correspond to a control genetic modification, or control genetic background? If so, Barb also names the genetic background/modification to which it should be compared.
18\tFalse\t15\t5,6,15,16,17\tThis sample received a control treatment\tDoes this sample correspond to a control **treatment** (other than genetic modification or background), for comparison with a different treatment in the same experiment? If so, Barb also names that different treatment.
19\tFalse\t16\t\tLow-level gene ontology terms\tLow-level gene ontology terms for biological processes Barb can infer for this experiment. Barb does not report generic processes such as histone or chromatin modification, or "Gene Regulation", "Gene expression", "Transcription", "Chromatin Accessibility", "Epigenetic regulation", "Remodeling", etc. and focuses instead on more specific processes such as "DNA damage repair", "Response to hypoxia", "Response to viral infection", "Brain development", etc
20\tFalse\t17\t19\tRelationship to COVID/pneumonia/inflammation/DNA damage\tIs this sample related to COVID/pneumonia/inflammation/DNA damage? (Barb answers as: Yes / No / Unsure)
"""

tasks: List[Task] = []
for line in task_data_v2_str.strip().split('\n'):
    if line.strip() == '':
        continue
    parts = line.split('\t')
    task_id: int = int(parts[0]) - 1
    forbid_number_prefix: bool = task_id == 11 or task_id == 12
    output_is_input_quote: bool = parse_bool_string(parts[1])
    previous_id: int = int(parts[2]) - 1
    dependent_ids: List[int] = [int(x) - 1 for x in parts[3].split(',')] if parts[3] else []
    n_generations: int = 10 if task_id == 0 or task_id == 10 or task_id == 11 or task_id == 12 or task_id == 13 else 3
    title: str = parts[4]
    details: str = parts[5]
    task = Task(id=task_id, previous_id=previous_id, dependent_ids=dependent_ids, title=title, details=details,  max_new_tokens=250 if parts[0] == '11' or parts[0] == '18' else 250, output_is_input_quote=output_is_input_quote, forbid_number_prefix=forbid_number_prefix, n_generations=n_generations)
    tasks.append(task)


barb_header: str = \
"""
Barb is a biologist analyzing metadata from a ChIP-seq experiment database. Her task is to extract information from\
 a record describing a single sample that is part of a larger study. The record may contain incomplete or misorganized\
 metadata, and it's Barb's job to identify the protein that was targeted in the ChIP experiment and to extract\
 information about the sample.

The record is:
```
"""

barb_footer: str = \
"""```

Barb parses all of the information above to complete the following (she outputs "N/A" or "Unsure" where appropriate). \
Unless a concise answer is requested, she thinks step by step and details her reasoning. Barb provides concise, \
professional, insightful, helpful, and truthful explanations for her answers.

"""

BARB_QA_EOL_MARKER: str = ''


class ShorteningSettings:
    def __init__(self, max_n_titles: int, summarize: bool):
        self.max_n_titles: int = max_n_titles
        self.summarize: bool = summarize
        self.summarize_everything: bool = False
    
    def try_harder(self) -> bool:  # return False if failed, because settings already were as tight at possible
        if self.max_n_titles > 0:
            self.max_n_titles = max(0, self.max_n_titles - 2)
            return True
        if not self.summarize:
            self.summarize = True
            return True
        if not self.summarize_everything:
            self.summarize_everything = True
            return True
        return False


def generate_barb_prompts(text_record: str,
                          tasks: List[Task],
                          group_id: str,
                          pre_existing_answers: List[str] = None,
                          max_tokens: int = 2048,
                          debug_files_base_name: str = '',
                          temperature: float = None,
                          do_sample: bool = True,
                          terminate_with_eos: bool = False,
                          ignore_dependent_questions: bool = False
                          ) -> Tuple[List[Dict], str]:  # prompts for all tasks, summarized record
    # preprocess text_record to update the intro and outro
    # deal with summarization if necessary
    previous_prompt_unused, first_expected_line, rest0 = extract_line_with_prefix(text_record, 'Sample name: ')
    beginning, last_expected_line, ignore = extract_line_with_prefix(first_expected_line + '\n' + rest0, 'Other info')
    if first_expected_line is None:
        raise ValueError('Did not find first line of record')
    if last_expected_line is None:
        raise ValueError('Did not find last line of record')
    actual_record: str = beginning + '\n' + last_expected_line
    all_titles, before_titles, after_titles = extract_titles_block(actual_record)
    shortening_settings: ShorteningSettings = ShorteningSettings(max_n_titles=15, summarize=False)
    shortenable_prompt_with_record: str = generate_prompt_up_to_QA(all_titles, before_titles, after_titles, shortening_settings=shortening_settings, debug_files_base_name='' if debug_files_base_name == '' else debug_files_base_name + '_a_', temperature=temperature, do_sample=do_sample)
    answers: List[str] = pre_existing_answers if pre_existing_answers is not None else extract_answers_from_barb_examples(text_record)
    result: List[Dict] = []
    iter: int = 0
    for task in tasks:
        full_prompt: str
        qa: str
        last_answer: str
        try:
            qa, last_answer = generate_barb_QA_chain(task, answers, ignore_dependent_questions)
        except Exception as e:
            raise ValueError(f'Problem with QA chain for {task} with {answers}', e)
        while True:
            iter += 1
            full_prompt = shortenable_prompt_with_record + qa
            unpadded_input_ids = tokenizer.encode(full_prompt, return_tensors="pt", padding="do_not_pad", max_length=max_tokens,
                                                  add_special_tokens=False).squeeze()
            extra_for_eos: int = 1 if terminate_with_eos else 0
            unpadded_length = len(unpadded_input_ids)
            if unpadded_length + (task.max_new_tokens if task.id >= len(answers) else 0) < max_tokens - extra_for_eos:
                padded_prompt: BatchEncoding = tokenizer(full_prompt, return_tensors="pt", padding="max_length",
                                                         return_attention_mask=True,  # probably not necessary
                                                         max_length=max_tokens,
                                                         add_special_tokens=False
                                                         )
                break
            tightened: bool = shortening_settings.try_harder()
            if not tightened:
                raise ValueError(f"Failed to bring size of prompt below max ({unpadded_length} >= {max_tokens} - {task.max_new_tokens} = {max_tokens - task.max_new_tokens} ; task_id={task.id} ; len(answers)={len(answers)}): {shortenable_prompt_with_record} \n\n\n-------\n\n\n Original was: {full_prompt}")
            shortenable_prompt_with_record: str = generate_prompt_up_to_QA(all_titles, before_titles, after_titles,
                                                                           shortening_settings=shortening_settings,
                                                                           debug_files_base_name='' if debug_files_base_name == '' else debug_files_base_name + '_b_' + str(iter),
                                                                           temperature=temperature,
                                                                           do_sample=do_sample)
        train_on_last_n: int = 0 if last_answer == '' else tokenized_length(last_answer, tokenizer) + 2
        labels: list[int] = []
        padded_prompt_with_eos: list[int] = []
        for i, t in enumerate(padded_prompt["input_ids"].squeeze().tolist()):
            if i < unpadded_length - train_on_last_n:
                labels.append(IGNORE_INDEX)
                padded_prompt_with_eos.append(t)
            elif i < unpadded_length:
                labels.append(t)
                padded_prompt_with_eos.append(t)
            elif terminate_with_eos and i == unpadded_length:
                labels.append(tokenizer.eos_token_id)
                padded_prompt_with_eos.append(tokenizer.eos_token_id)
            else:
                labels.append(IGNORE_INDEX)
                padded_prompt_with_eos.append(t)
        if terminate_with_eos:
            if padded_prompt["attention_mask"][0][unpadded_length] != 0 or padded_prompt["attention_mask"][0][unpadded_length - 1] != 1:
                raise ValueError(f'Unexpected value for attention mask {padded_prompt["attention_mask"][0][unpadded_length]}')
            padded_prompt["attention_mask"][0][unpadded_length] = 1
        result.append({
            "unencoded_text": full_prompt,
            "input_ids": padded_prompt_with_eos,  # The output pt tensor has a useless `batch` dimension
            "labels": labels,
            "attention_mask": padded_prompt["attention_mask"].squeeze().tolist(),
            "group_id": group_id
        })
    return result, shortenable_prompt_with_record


PROTOCOL_PARAGRAPH_HEADER: str = 'The protocol information in this paragraph likely'


def generate_prompt_up_to_QA(all_titles: List[str], before_titles: str, after_titles: str, shortening_settings: ShorteningSettings, debug_files_base_name: str = None, temperature: float = None, do_sample: bool = None) -> str:
    if len(all_titles) > shortening_settings.max_n_titles:
        if shortening_settings.max_n_titles == 0:
            all_titles = []
        else:
            all_titles = all_titles[0:max(1, shortening_settings.max_n_titles)]  # First line has sample name for present record; must keep it
            all_titles[0] = 'Titles of *some* samples in the study:'
            all_titles.append('    - ...')
    if shortening_settings.summarize and not shortening_settings.summarize_everything:
        empty_after_titles: bool = after_titles.strip() == ''
        protocol_split = after_titles.split(PROTOCOL_PARAGRAPH_HEADER) if not empty_after_titles else before_titles.split(PROTOCOL_PARAGRAPH_HEADER)
        if len(protocol_split) == 1:
            logger.log(3, f'ONLY 1 line in {after_titles}')
            pass
        elif len(protocol_split) == 2:
            part_2: str = PROTOCOL_PARAGRAPH_HEADER + protocol_split[1]
            lines = part_2.split('\n', maxsplit=1)
            if len(lines) != 2:
                raise ValueError(f'Unexpected line split {lines}')
            summarized: Optional[str] = None
            try:
                summarized: str = bob_summarize(lines[0], temperature=temperature, debug_files_base_name=debug_files_base_name, do_sample=do_sample)
            except Exception:
                logger.log(3, 'IGNORING SUMMARIZING ERROR')
                summarized = lines[0]
            if empty_after_titles:
                if protocol_split[0] is None:
                    raise ValueError('protocol_split[0] is None')
                if summarized is None:
                    raise ValueError('summarized is None')
                if lines[1] is None:
                    raise ValueError('lines[1] is None')
                before_titles = protocol_split[0] + summarized + lines[1]
            else:
                after_titles = protocol_split[0] + summarized + lines[1]
        else:
            raise ValueError('Impossible split ' + str(protocol_split))
    elif shortening_settings.summarize_everything:
        if before_titles.strip() != '':
            try:
                before_titles = '\n'.join([bob_summarize(l, temperature=temperature, do_sample=do_sample, debug_files_base_name='' if debug_files_base_name == '' else debug_files_base_name + 'before_titles') for l in before_titles.split('\n')])
            except Exception as e:
                logger.log(3, f'IGNORING SUMMARIZING ERROR {e}')
        if after_titles.strip() != '':
            try:
                after_titles = '\n'.join([bob_summarize(l, temperature=temperature, do_sample=do_sample, debug_files_base_name='' if debug_files_base_name == '' else debug_files_base_name + 'after_titles') for l in after_titles.split('\n')])
            except Exception as e:
                logger.log(3, f'IGNORING SUMMARIZING ERROR {e}')
    final_prefix: str = barb_header + '\n'.join(
        [before_titles, '\n'.join(all_titles), after_titles]).rstrip('\n .') + '.\n' + barb_footer
    return final_prefix


def generate_barb_QA_chain(task: Task, answers: List[str], ignore_dependent_questions: bool = False) -> Tuple[str, str]:
    result: str = ''
    last_answer: str = ''
    if not ignore_dependent_questions:
        for dependent_task in [tasks[id] for id in task.dependent_ids]:
            if answers[dependent_task.id] == '':
                raise ValueError(f"Dependent task {dependent_task.title} should have been performed before {task.title}")
            result += dependent_task.title + ': ' + answers[dependent_task.id] + BARB_QA_EOL_MARKER + '\n\n'
    result += task.details + ': '
    if len(answers) > task.id:  # We already have the answer to the question; we're generating a prompt for training
        last_answer = answers[task.id] + BARB_QA_EOL_MARKER + '\n'
        result += last_answer
    return result, last_answer


def process_Barb_training_directory(directory_path: str) -> List[List[str]]:  # return a list of texts that include all answers
    if not os.path.isdir(directory_path):
        raise ValueError(f"The specified directory '{directory_path}' does not exist.")
    
    result: List[List[str]] = []
    for filename in os.listdir(directory_path):
        if filename[0] == '.':
            continue
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            processed_lines = process_Barb_training_file(file_path)
            result.append(processed_lines)
            logger.debug(f"Processed lines for file '{filename}':")
            logger.debug(processed_lines)
    return result


def barb_training_directory_to_dataset(directory_path: str) -> List[List[Dict]]:  # return a list of texts that include all answers
    if not os.path.isdir(directory_path):
        raise ValueError(f"The specified directory '{directory_path}' does not exist.")
    result: List[List[Dict]] = []
    for filename in os.listdir(directory_path):
        if filename[0] == '.':
            continue
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                tokenized, summarized_prompt =\
                    generate_barb_prompts(open(file_path).read(), tasks, group_id=filename,
                                                       terminate_with_eos=True, debug_files_base_name='debug_summarize_' + filename)
                result.append(tokenized)
            except ValueError as e:
                logger.error(f'**** SKIPPING SAMPLE {file_path} BECAUSE OF ERROR {e}')
    return result


def process_Barb_training_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        try:
            return extract_answers_from_barb_examples(file.read())
        except ValueError as e:
            raise ValueError(f"Problem with file {file}") from e


def extract_barb_answers_starting_from_0(string: str) -> List[str]:
    answer_list: List[str] = []
    found_lines_with_0_prefix: int = 0
    number_of_0dots: int = count_lines_starting_with_zero_dot(string)
    if number_of_0dots < 1 or number_of_0dots > 2:
        raise ValueError(f'Expected 1 or two lines starting with 0. but found {number_of_0dots} in {string}')
    next_expected_number: int = -1
    found_number: int = -1
    for line in string.split('\n'):
        line = line.strip()
        if found_lines_with_0_prefix < number_of_0dots:
            if line.startswith('0. '):
                found_lines_with_0_prefix += 1
        if found_lines_with_0_prefix == number_of_0dots:
            match: Match = re.match(r'^(\d+)\.\s', line)
            if match:
                next_expected_number += 1
                found_number = int(match.group(1))
                if found_number != next_expected_number:
                    raise ValueError(f'Unexpected number {found_number} instead of {next_expected_number} while parsing {string}')
                rest_of_line: str = line[line.find('.') + 1:]
                rest_of_line = rest_of_line.strip(' .\t')
                rest_of_line = rest_of_line.replace('\\:', ':')
                answer_list.append(rest_of_line)
                if found_number == len(tasks) - 1:
                    logger.debug(f'Extracted answers 2: {answer_list}')
                    return answer_list
    raise ValueError(f'Should not reach here; last read answer #{found_number} found {answer_list} from {string}')


def extract_answers_from_barb_examples(string: Union[str, TextIO]) -> List[str]:
    # Check if there is a line that starts with 0.
    found_line_with_0_prefix: bool = False
    for line in string.split('\n'):
        if line.strip().startswith('0. '):
            found_line_with_0_prefix = True
    if found_line_with_0_prefix:
        return extract_barb_answers_starting_from_0(string)
    raise Exception('Should not reach here')


def extract_titles_block(text: str) -> tuple[list[str], str, str]:
    lines = text.split('\n')
    
    titles_block = []
    before_titles = []
    after_titles_block = []
    
    in_titles_block = False
    titles_block_ended = False
    
    titles_found = False
    
    for line in lines:
        if line.startswith("Titles"):
            if titles_found:
                raise ValueError(f"Input contains multiple 'Titles' lines: {text}")
            titles_block.append(line)
            in_titles_block = True
            titles_found = True
        elif line.startswith("    -") and in_titles_block:
            titles_block.append(line)
        elif not in_titles_block and not titles_block_ended:
            before_titles.append(line)
        elif in_titles_block:
            in_titles_block = False
            titles_block_ended = True
            after_titles_block.append(line)
        elif not in_titles_block and titles_block_ended:
            after_titles_block.append(line)
    
    before_titles = '\n'.join(before_titles)
    after_titles_block = '\n'.join(after_titles_block)
    
    return titles_block, before_titles, after_titles_block


def extract_line_with_prefix(s, p) -> tuple[str, Optional[str], str]:
    lines = s.split('\n')
    
    line_with_prefix = None
    before_line = []
    after_line = []
    
    line_found = False
    
    for line in lines:
        if line.startswith('---'):
            break
        if line.startswith(p):
            if line_found:
                raise ValueError(f"Input contains multiple lines with prefix '{p}': ${s}")
            line_with_prefix = line
            line_found = True
        elif not line_found:
            before_line.append(line)
        elif line_found:
            after_line.append(line)
    
    before_line = '\n'.join(before_line)
    after_line = '\n'.join(after_line)
    
    # Return the results
    return before_line, line_with_prefix, after_line


def bob_summarize0(input: str, debug_files_base_name: str = None, temperature: float = None, do_sample: bool = None) -> str:
    input = input.strip('\n \t.')
    if input == '':
        logger.debug('Skipping summarization of blank string')
        return ''
    if input.startswith('Sample name'):
        logger.debug('Not summarizing sample name')
        return input
    if len(input) < 600:
        logger.debug('Not summarizing short input')
        return input
    prompt: str = shortened_bob_prompt_base + '\n' + input + '\n```'
    sentences: List[tuple[str, bool, float]]
    sentences, _ = process_abstract(prompt, temperature=temperature, do_sample=False if do_sample is None else do_sample, debug_files_base_name=debug_files_base_name)
    return join_strings_with_period([s for s, keep, conf in sentences if keep])


def bob_summarize(input: str, debug_files_base_name: str = None, temperature: float = None, do_sample: bool = None) -> str:
    return summarizations.__getitem__(input)


summarizations: AutoComputedShelfDB = AutoComputedShelfDB('summaries_cache', bob_summarize0)


def save_token_probabilities(file_path: str, details, tokenizer: PreTrainedTokenizer):
    sequences = details['sequences']
    scores = details['scores']
    input_length = sequences.size(1) - len(scores)
    generated_tokens = sequences[0, input_length:].tolist()
    generated_token_probs = []
    for logit, token_id in zip(scores, generated_tokens):
        probabilities = softmax(logit, dim=1)
        token_prob = probabilities[0, token_id].item()
        generated_token_probs.append(token_prob)
    generated_words = tokenizer.convert_ids_to_tokens(generated_tokens)
    with open(file_path, 'w', encoding='utf-8') as file:
        unused = file.write("Start_Index\tToken\tProbability\n")
        for index, word in enumerate(generated_words):
            unused = file.write(f"{index}\t{word}\t{generated_token_probs[index]}\n")


class NoProcessedSentence(BaseException):
    pass


class ListTooShort(BaseException):
    pass


LAST_SENTENCE_MARKER: str = "We used siRNA to knock down Notch1"


def extract_positive_sentences(model,
                               tokenizer: PreTrainedTokenizer,
                               details: Dict[str, Any] = None,  # optional pre-existing model generation
                               prompt: str = None,  # if details==None, prompt for first run
                               write_prompt_iterations_to_file: str = '',
                               kept_sentences: List[Tuple[str, bool, float]] = None,
                               completion_with_confidence: str = '',
                               completion_without_confidence: str = '',
                               temperature: float = 0.1,
                               do_sample: bool = False)\
        -> Tuple[List[Tuple[str, bool, float]], str, str, str]:  # List of positive sentences, text with annotations, text without, last fully processed sentence
    if details is None:
        if write_prompt_iterations_to_file != '':
            with open(write_prompt_iterations_to_file + "_prompt", 'w', encoding='utf-8') as file:
                file.write(prompt)
        outtext_unused, details = generate_from_lora(prompt, model, tokenizer, temperature=temperature, do_sample=do_sample)
        if write_prompt_iterations_to_file != '':
            with open(write_prompt_iterations_to_file + "_outtext", 'w', encoding='utf-8') as file:
                file.write(outtext_unused)
    elif prompt is None:
        raise ValueError('Unused code path')
    else:
        raise ValueError("Should not provide both details and prompt")
    if kept_sentences is None:
        kept_sentences = []
    sequences = details['sequences']
    scores = details['scores']
    if len(scores) > 1700:
        raise ValueError(f'Cannot have generated {len(scores)} tokens with input string length {len(prompt)} and sequences.size {sequences.size(1)} ')
    input_length: int = sequences.size(1) - len(scores)
    generated_tokens: List[int] = sequences[0, input_length:].tolist()
    all_text: str = tokenizer.decode(sequences[0], skip_special_tokens=True)
    local_block_tokens: List[int] = []
    lowest: float = float('inf')
    second_lowest: float = float('inf')
    last_processed_sentence: str = ''
    previous_last_processed_sentence: Optional[str] = None
    for logit, token_id in zip(scores, generated_tokens):
        probabilities = softmax(logit, dim=1)
        token_prob = probabilities[0, token_id].item()
        if token_prob < lowest:
            second_lowest = lowest
            lowest = token_prob
        elif token_prob < second_lowest and token_prob != lowest:
            second_lowest = token_prob
        local_block_tokens.append(token_id)
        local_text_block: str = tokenizer.decode(local_block_tokens, skip_special_tokens=True)
        if local_text_block.endswith("###END"):
            local_text_block = local_text_block.strip()
            if not local_text_block.startswith('Sentence'):
                local_text_block = 'Sentence:\n' + local_text_block
            lines: List[str] = [l.strip() for l in local_text_block.splitlines(keepends=False) if not l.strip() == '']
            if len(lines) < 6:
                new_lines: List[str] = []
                for l in lines:
                    if l.startswith("Bob's explanation") and len(l) > len("Bob's explanation") + 4:
                        new_lines.append("Bob's explanation")
                        new_lines.append(l[len("Bob's explanation:"):].strip())
                    elif l.startswith("Sentence") and len(l) > len("Sentence:") + 4:
                        new_lines.append("Sentence:")
                        new_lines.append(l[len("Sentence:"):].strip())
                    else:
                        new_lines.append(l)
                        if len(new_lines) > 6:
                            raise ValueError(f'Too many lines, starting from {lines} \nand getting to: {new_lines}')
                lines = new_lines
                local_text_block = '\n'.join(new_lines)  # Will be used for saved completion
                if len(lines) != 6:
                    raise ListTooShort(f"List too short ({len(lines)}): {lines}; text block was {local_text_block} with prompt {prompt}")
            dont_discard_line: bool = False
            if len(lines) > 6:
                #  Maybe the input chunk was split over many lines; check if we can rescue it
                if lines[0].startswith('Sentence') and lines[-4].startswith("Bob's explanation"):
                    new_lines = ['Sentence:', join_strings_with_dots(lines[1:-4]), lines[-4], lines[-3], lines[-2], lines[-1]]
                    logger.debug(f'Fixed up {lines} to {new_lines}')
                    dont_discard_line = True  # Joining with periods might mistakenly make us think the sentence was hallucinated
                    lines = new_lines
                else:
                    raise ValueError(f"Too many lines: {lines}")
            previous_last_processed_sentence = last_processed_sentence
            last_processed_sentence = lines[-5].strip(' \t.')  # Stripping may be unnecessary here
            logger.debug(f'Fully processed {last_processed_sentence}')
            if last_processed_sentence.replace('°', '_') not in prompt.replace('°', '_') and not dont_discard_line \
                    and LAST_SENTENCE_MARKER in last_processed_sentence:
                last_processed_sentence = last_processed_sentence.replace(LAST_SENTENCE_MARKER, '')
                last_processed_sentence = last_processed_sentence.strip(' \t.')
                if last_processed_sentence == '':
                    break
            if last_processed_sentence.replace('°', '_') not in prompt.replace('°', '_') and not dont_discard_line:
                # The model seems to [at least sometimes] change ° to _
                logger.debug(f'Discarding probably hallucinated sentence {last_processed_sentence}')
                last_processed_sentence = previous_last_processed_sentence
            else:
                if len(last_processed_sentence) < 5:
                    logger.debug(f"Warning: very short sentence {lines}")
                if "yes" in lines[-1].lower() or second_lowest < 0.5:
                    kept_sentences.append((last_processed_sentence, "yes" in lines[-1].lower().replace("2:yes", ""), second_lowest))
                    completion_without_confidence = completion_without_confidence + local_text_block + '\n\n'
                completion_with_confidence = (completion_with_confidence + local_text_block + '\n'
                                        + f'Confidence1:{lowest:.1f}\nConfidence2:{second_lowest:.1f}\n\n')
            lowest: float = float('inf')
            second_lowest: float = float('inf')
            local_block_tokens = []
    if last_processed_sentence is None:
        raise NoProcessedSentence
    if last_processed_sentence.strip('. \t\n') == LAST_SENTENCE_MARKER:
        if len(kept_sentences) > 0:
            s, _, _ = kept_sentences[-1]
            if s == last_processed_sentence:
                kept_sentences.pop()
        last_processed_sentence = previous_last_processed_sentence
    return kept_sentences, completion_with_confidence, completion_without_confidence, last_processed_sentence


def gcGPU():
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated()} bytes")


def load_lora_checkpoint(lora_bin_path: str, **kwargs):
    adapter_weights = torch.load(lora_bin_path)
    model = lora_model_init(**kwargs)  # TODO retrieve LoRa parameters from saved JSON file
    model = peft.set_peft_model_state_dict(model, adapter_weights)
    return model


class StopStringsCriterionSub(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, stop_strings: List[Tuple[str, int, int, int, str]], min_length: int = -1):
        super().__init__()
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.stop_strings: List[Tuple[str, int, int, int, str]] = stop_strings
        self.min_length = min_length
    
    def __call__(self, input_ids, scores = None, **kwargs) -> bool:
        if self.min_length != -1 and input_ids.shape[-1] < self.min_length:
            return False
        for i, (stop_string, encoded_length, max_encounters, current_count, next_stop) in enumerate(self.stop_strings):
            if self.min_length == -1:
                last_tokens = input_ids[:, -encoded_length:]
            else:
                last_tokens = input_ids[:, self.min_length + 1:]
                last_tokens = last_tokens[:, -encoded_length:]
            decoded_text: str = self.tokenizer.decode(last_tokens[0])
            index: int = decoded_text.find(stop_string)
            if index != -1:
                current_count += 1
                if current_count >= max_encounters:
                    if next_stop != '':
                        self.stop_strings = [(next_stop, tokenized_length(next_stop, self.tokenizer), 1, 0, '')]
                        return False
                    else:
                        logger.debug( f'Stopping generation at {self.tokenizer.decode(input_ids[:, -150:][0])}')
                        return True
                self.stop_strings[i] = stop_string, encoded_length, max_encounters, current_count, next_stop
        return False


def replace_second_occurrence_add_dot(s: str, old: str, new: str) -> str:
    first_index: int = s.find(old)
    
    if first_index != -1:
        second_index = s.find(old, first_index + 1)
        
        if second_index != -1:
            return join_strings_with_period([s[:second_index].rstrip('\n .'), new]) + s[second_index + len(old):]
    
    return s


# Returns: [generated_text, perplexity, perplexity of truncated generated_text, max_token_p]
def generate_alternatives(model, tokenizer, prompt: str, max_length: int = 2048, max_new_tokens: int = None, max_generations: int = 10, threshold: float = 0.7,
                          top_p: float = 0.92, top_k: int = 12, temperature: float = 1.0, do_sample: bool = True, num_beams: int = 1, num_beam_groups: int = 1,
                          stopping_criteria: StoppingCriteriaList = None, ban_number_prefix: bool = False, low_mem: int = False) -> List[Tuple[str, float, float, float]]:
    # Initialize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    outputs: list[tuple[Tensor, list[float], Any]] = [(input_ids, [], None)]
    
    model.eval()
    
    stopping_criterion = StopStringsCriterionSub(tokenizer, [
        initial_stopping_criteria_tuple
    ])
    
    if stopping_criteria is None:
        stopping_criteria = StoppingCriteriaList([stopping_criterion])
    
    eos_token_id = model.config.eos_token_id
    
    n_variants: int = 0
    
    with torch.no_grad():
        with torch.autocast("cuda"):
            l: int = -1
            for i in range(max_length - input_ids.size(1)):
                if n_variants == max_generations:
                    break
                l = i
                new_outputs: list[tuple[Tensor, list[float], Any]] = []
                while outputs:
                    output_head = outputs.pop(0)
                    output, probabilities, past_key_values = output_head
                    if output[0, -1].item() == eos_token_id:
                        new_outputs.append((output, probabilities, past_key_values))
                        continue
                    
                    # TODO: to use past_key_values if it's not None, only feed the model the last-generated token
                    model_out = model(output if past_key_values is None else torch.tensor(output[0, -1].item()).unsqueeze(dim=0).unsqueeze(dim=0).to('cuda'),
                                      use_cache=True, past_key_values=None if past_key_values is None else
                        tuple((kv[0].to('cuda'), kv[1].to('cuda')) for kv in past_key_values))
                    logits = model_out.logits
                    if (low_mem > 0) and len(new_outputs) < 2 and len(outputs) < 2:
                        new_past_key_values = model_out.past_key_values
                    else:
                        new_past_key_values = tuple((kv[0].to('cpu'), kv[1].to('cpu')) for kv in model_out.past_key_values)
                    
                    scaled_logits = logits[:, -1, :] / temperature
                    
                    token_probs = torch.softmax(scaled_logits, dim=-1)
                    
                    best_prob, best_idx = torch.max(token_probs, dim=-1)
                    
                    next_token = best_idx
                    
                    concat = torch.cat((output, next_token.unsqueeze(0).to('cuda')), dim=1)
                    
                    while ban_number_prefix and i < 3:
                        next_token_string = tokenizer.decode(concat[0][-min(2, (i + 1)):], skip_special_tokens=True)
                        if not next_token_string.strip().isnumeric():
                            break
                        logger.debug(f'Suppressing number {next_token_string} with prob {best_prob} after {tokenizer.decode(concat[0][-50:], skip_special_tokens=True)}')
                        token_probs[0][best_idx] = 0
                        best_prob, best_idx = torch.max(token_probs, dim=-1)
                        next_token = best_idx
                        concat = torch.cat((output, next_token.unsqueeze(0).to('cuda')), dim=1)
                    
                    new_output = concat
                    
                    stop_condition: bool = False
                    for stopping_criterion in stopping_criteria:
                        if stopping_criterion(new_output):
                            new_output = torch.cat((new_output, torch.tensor([[eos_token_id]]).to('cuda')), dim=1)
                            stop_condition = True
                            break
                    new_outputs.append((new_output, probabilities + [best_prob.item()], new_past_key_values))
                    
                    if stop_condition:
                        # Favor shorter outputs; don't look for variants
                        continue
                    
                    # If the best token has a probability lower than the threshold, also consider the second-best token
                    if best_prob < threshold and n_variants < max_generations:
                        n_variants += 1
                        _, sorted_indices = torch.sort(token_probs, descending=True)
                        second_best_idx = sorted_indices[0][1]
                        next_token = second_best_idx
                        concat = torch.cat((output, next_token.unsqueeze(0).unsqueeze(0).to('cuda')), dim=1)
                        
                        while ban_number_prefix and i < 3:
                            next_token_string = tokenizer.decode(concat[0][-min(2, (i + 1)):], skip_special_tokens=True)
                            if not next_token_string.strip().isnumeric():
                                break
                            token_probs[0][second_best_idx] = 0
                            _, sorted_indices = torch.sort(token_probs, descending=True)
                            second_best_idx = sorted_indices[0][1]
                            next_token = second_best_idx
                            concat = torch.cat((output, next_token.unsqueeze(0).unsqueeze(0).to('cuda')), dim=1)
                        
                        new_output = concat
                        
                        for stopping_criterion in stopping_criteria:
                            if stopping_criterion(new_output):
                                new_output = torch.cat((new_output, torch.tensor([[eos_token_id]]).to('cuda')), dim=1)
                                break
                        
                        second_best_prob = token_probs[0, second_best_idx].item()
                        new_outputs.append((new_output, probabilities + [second_best_prob], new_past_key_values))
                
                outputs = new_outputs
            generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k,
                                                 num_beams=num_beams, num_beam_groups=num_beam_groups,
                                                 do_sample=do_sample, eos_token_id=model.config.eos_token_id,
                                                 pad_token_id=model.config.pad_token_id)
            
            generate_params: Dict[str, Any] = {
                "generation_config": generation_config,
                "return_dict_in_generate": True,
                "output_scores": True,  # num_beams == 1,
                "output_attentions": False,
                "output_hidden_states": False,
                "stopping_criteria": stopping_criteria,
                "use_cache": True,
            }
            if max_new_tokens is not None:
                generate_params['max_length'] = min(max_length, len(prompt) + l + max_new_tokens)
            else:
                generate_params['max_length'] = max_length
            if low_mem:
                gcGPU()
            new_outputs: List[Tuple[str, float, float, float]] = []
            while outputs:
                output, probabilities, past_key_values = outputs.pop()
                seq = None
                if output[0, -1].item() == eos_token_id:
                    new_out = {"input_ids": torch.empty(0), "labels": torch.empty(0), "scores": []}
                    seq = output[0]
                else:
                    to_cuda = output.to('cuda')
                    if past_key_values is None or low_mem > 1:
                        generate_params['past_key_values'] = None
                    else:
                        generate_params['past_key_values'] = tuple((kv[0].to('cuda'), kv[1].to('cuda')) for kv in past_key_values)
                    generate_params['input_ids'] = to_cuda
                    new_out = model.generate(**generate_params)
                    seq = new_out['sequences'][0]
                perplexity, min_p = compute_perplexity(new_out['scores'],
                                   seq[-len(new_out['scores']):], prefix_probabilities=probabilities)
                short_perplexity, _ = compute_perplexity(new_out['scores'],
                                                         seq[-len(new_out['scores']):],
                                                         prefix_probabilities=probabilities,
                                                         limit_to_first_n=10)
                generate_params['past_key_values'] = None
                generate_params['input_ids'] = None
                del new_out
                gcGPU()
                new_outputs.append((tokenizer.decode(seq, skip_special_tokens=True), perplexity, short_perplexity, min_p))
    
    return new_outputs


@use_defaults_on_none
def generate_from_lora(input_string: str, model, tokenizer: PreTrainedTokenizer, max_n_sentences: int = 100,
                       temperature: float = 0.1, top_p: float = 0.92, top_k: int = 12, num_beams: int = 1, num_beam_groups: int = 1, do_sample: bool = False,
                       add_last_sentence_marker: bool = True, max_new_tokens: int = None, stopping_criteria: StoppingCriteriaList = None) -> \
        tuple[str, any]:  # return string output as well as full model.generate output (when sampling, generation_utils.SampleDecoderOnlyOutput)
    if num_beam_groups > 1 and do_sample:
        logger.debug("Won't work, for now; forcing num_beam[groups] to 1")
        num_beam_groups = 1
        num_beams = 1
    if add_last_sentence_marker and input_string.find(LAST_SENTENCE_MARKER) == -1:
        logger.debug("Adding last sentence marker")
        input_string = replace_second_occurrence_add_dot(input_string, '\n```', LAST_SENTENCE_MARKER + '.\n```')
    
    if stopping_criteria is None:
        stopping_criteria = StoppingCriteriaList([StopStringsCriterionSub(tokenizer, [
                                                                                  initial_stopping_criteria_tuple
                                                                                 ])])
    logger.debug(f'Temperature: {temperature}; sampling: {do_sample}; top_p: {top_p}; top_k:{top_k}')
    generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k,
                                         num_beams=num_beams, num_beam_groups=num_beam_groups,
                                         do_sample=do_sample, eos_token_id=model.config.eos_token_id,
                                         pad_token_id=model.config.pad_token_id)
    encoded_input = tokenizer.encode(input_string)
    input_tensor = torch.tensor(encoded_input).unsqueeze(0).to('cuda')
    generate_params: Dict[str, Any] = {"input_ids": input_tensor,
                                       "labels": input_tensor,
                                   "generation_config": generation_config,
                                   "return_dict_in_generate": True,
                                   "output_scores": True,
                                    "output_attentions": False,
                                    "output_hidden_states": False,
                                   "stopping_criteria": stopping_criteria,
                                   "use_cache": True,
                                   }
    if max_new_tokens is not None:
        generate_params['max_length'] = min(2048, len(encoded_input) + max_new_tokens)
    else:
        generate_params['max_length'] = 2048
    with torch.no_grad():
        with torch.autocast("cuda"):
            with torch.inference_mode():  # Makes model.eval below and no_grad above redundant?
                model.eval()
                output = model.generate(**generate_params)
        return tokenizer.decode(output.sequences[0], skip_special_tokens=True), output


@use_defaults_on_none
def do_all_barb_tasks(input_string: str, tasks: Sequence[Task], model, tokenizer: PreTrainedTokenizer,
                      temperature: float = 0.1, top_p: float = 0.92, top_k: int = 12, num_beams: int = None, num_beam_groups: int = None, do_sample: bool = True,
                      debug: bool = False,
                      ignore_dependent_questions: bool = False,
                      max_new_tokens: int = None) -> Tuple[List[Tuple[List[str], List[float], float]],  # answer_text, perplexity, prompt perplexity (latter currently not working: maxes out GPU VRAM for some weird reason)
                                                           str]:  # shortest summary generated
    answers: List[Tuple[List[str], List[float], float]] = []
    answers_string_only: List[str] = []
    model.eval()
    summarized_before_qa_shortest: Optional[str] = None
    for i, task in enumerate(tasks):
        prompts, summarized_before_qa = generate_barb_prompts(input_string, [task], 'unused',
                                                              pre_existing_answers=answers_string_only,
                                                              ignore_dependent_questions=ignore_dependent_questions)
        prompt: Dict[str, Any] = prompts[0]
        if summarized_before_qa_shortest is None or len(summarized_before_qa) < len (summarized_before_qa_shortest):
            summarized_before_qa_shortest = summarized_before_qa
        prompt_perplexity: float = 0
        length_encoded_prompt = len(tokenizer.encode(prompt['unencoded_text'], padding="do_not_pad"))
        tries: List[Tuple[str, float, float, float, float]] = []
        stopping_criteria = StoppingCriteriaList([StopStringsCriterionSub(tokenizer, [
            ('\n', 1, 1, 0, '')
        ], min_length=length_encoded_prompt + 1)])
        alternatives: List[Tuple[str, float]] = []
        threshold_p: float = 0.7
        low_mem: int = 0
        while len(alternatives) < 5 and threshold_p < 0.99:
            while True:
                try:
                    alternatives: List[Tuple[str, float, float, float]] = generate_alternatives(
                        model=model, tokenizer=tokenizer, prompt=prompt['unencoded_text'], max_generations=task.n_generations,
                        max_new_tokens=task.max_new_tokens,
                        temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, num_beam_groups=num_beam_groups, do_sample=do_sample,
                        stopping_criteria=stopping_criteria,
                        threshold=threshold_p,
                        ban_number_prefix=task.forbid_number_prefix,
                        low_mem=low_mem
                    )
                except torch.cuda.OutOfMemoryError as e:
                    # `del` of *any* variable required for subsequent garbage completion to free up GPU memory???
                    del alternatives
                    alternatives = []
                    gcGPU()
                    if low_mem < 2:
                        low_mem += 1
                        continue
                    else:
                        raise RuntimeError('Cuda out of memory', e)
                break
            sorted_alternatives = sorted(alternatives, key=lambda x: x[1])  # Sort by answer perplexity
            if sorted_alternatives[0][1] <= 0.0021:
                break
            # stop if min token probability in lowest-perplexity answer is sufficiently high
            if sorted_alternatives[0][3] >= 0.9:
                break
            threshold_p += 0.1
        for out_text, perplexity, short_perplexity, min_token_p in alternatives:
            answer: str = out_text[len(prompt['unencoded_text']):]
            if answer.startswith('\n\n'):
                stripped: str = answer.strip()
                if '?' in stripped or ':' in stripped:  # Probably hallucinating another question
                    logger.debug('Probable question hallucination after empty answer; forcing empty answer')
                    answer = ''
            answer = answer.strip()
            if '\n' in answer:
                raise ValueError(f'Unexpected newline in answer to question {task.title}: XXX{answer}XXX')
            if answer == '':
                perplexity = 100
                short_perplexity = 100
            tries.append((answer, perplexity, short_perplexity, prompt_perplexity, min_token_p))
        sorted_tries = sorted(tries, key=lambda x: x[1])  # Sort by answer perplexity
        logger.debug(f'Answers and perplexities: {sorted_tries}')
        best_answer = sorted_tries[0]
        best_answer_text: str = best_answer[0]
        if best_answer_text == '':
            best_answer_text = 'N/A'
        reported_answer_texts: List[str] = []
        reported_answer_perxs: List[float] = []
        for a in sorted_tries:
            if a[0] not in reported_answer_texts:
                reported_answer_texts.append(a[0])
                reported_answer_perxs.append(a[1])
        answers.append((reported_answer_texts, reported_answer_perxs, best_answer[2]))
        answers_string_only.append(best_answer_text)
    return answers, summarized_before_qa_shortest


answer_perplexities: shelve.Shelf[List[float]] = shelve.open('answer_perplexities', writeback=True)


def process_all_files_in_directory(model,
                                   input_directory_path: str,
                                   output_directory_path: str,
                                   high_perplexity_directory: str,
                                   temperature: float = 0.5,
                                   ignore_dependent_questions: bool = False
                                   ) -> None:
    try:
        os.makedirs(output_directory_path, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Error creating directory {output_directory_path}: {e}", e)
    try:
        os.makedirs(high_perplexity_directory, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Error creating directory {high_perplexity_directory}: {e}", e)
    
    txt_files: List[str] = [file for file in os.listdir(input_directory_path) if file.endswith('.txt')]
    for file_name in txt_files:
        output_file: str = output_directory_path + '/' + file_name[:-4] + '_answers.txt'
        if os.path.exists(output_file):
            logger.debug(f'Skipping {file_name} because output already exists')
            continue
        try:
            with time_limit(3000):
                with open(input_directory_path + '/' + file_name, 'r') as f:
                    text: str = f.read()
                answers: List[Tuple[List[str], List[float], float]]
                out_text, answers, shortest_summary_generated = process_record(text, model=model,
                                                                               temperature=temperature,
                                                                               num_beams=1,
                                                                               num_beam_groups=1,
                                                                               ignore_dependent_questions=ignore_dependent_questions)
                with open(output_file, 'w') as out_file:
                    out_file.write(out_text)
                    out_file.write('\n-----------------\n')
                    out_file.write(shortest_summary_generated)
                high_perplexity_tasks: List[int] = []
                for i, answer in enumerate(answers):
                    l: List[float] = answer_perplexities.get(str(i), [])
                    index = bisect.bisect_left(l, answer[1][0])
                    l.insert(index, answer[1][0])
                    answer_perplexities[str(i)] = l
                    if index / len(l) > 0.95:  # High perplexity answer
                        high_perplexity_tasks.append(i)
                answer_perplexities.sync()
                if len(high_perplexity_tasks) > 0:
                    with open(f'{high_perplexity_directory}/{"_".join([str(k) for k in high_perplexity_tasks])}_{file_name}', 'w') as f:
                        f.write(out_text)
                        f.write('\n-----------------\n')
                        f.write(f'{high_perplexity_tasks}\n')
                        f.write(f'{answer_perplexities}')
                        f.write(shortest_summary_generated)
        except TimeoutException as timeout:
            logger.error(f'Timed out on {file_name}')
        except (ValueError, KeyError, NameError, TypeError) as ve:
            error_message: str = f'Exception while processing {file_name}: {ve}\n{traceback.format_exc()}'
            logger.error(error_message)
            with open(output_directory_path + '/' + file_name[:-4] + '.error', 'w') as err_file:
                err_file.write(error_message)


def process_record(text: str, model,
                   temperature: float = 0.1, top_p: float = 0.92, top_k: int = 12, num_beams: int = None, num_beam_groups: int = None, do_sample: bool = True,
                   ignore_dependent_questions: bool = False)\
                   -> Tuple[str, List[Tuple[List[str], List[float], float]], str]:
    answers: List[Tuple[List[str], List[float], float]] = []
    shortest_summary_generated: str = ''
    answers, shortest_summary_generated = do_all_barb_tasks(input_string=text, tasks=tasks, model=model, tokenizer=tokenizer,
                                temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
                                num_beam_groups=num_beam_groups, do_sample=do_sample, # noqa E127
                                ignore_dependent_questions=ignore_dependent_questions
                                )
    end_record_index: int = find_nth_occurrence('```', text, 2)
    if end_record_index == -1:
        end_record_index = text.find('Now parse all of the information above to complete ')
        if end_record_index == -1:
            raise ValueError(f'Missing end of block in {text}')
        end_record_index -= 5
    answer_text = text[:end_record_index + 4] + '\n\n'
    for i, a in enumerate(answers):
        answer_text += f'{i}. {a[0][0]}\n     p={a[1][0]:.3f}\n'
        for j in range(1, len(a[0])):
            answer_text += f'     p={a[1][j]:.3f}      {a[0][j]}\n'
    return answer_text, answers, shortest_summary_generated


def compute_perplexity(logits: List[torch.Tensor], generated_tokens: torch.Tensor, prefix_probabilities: List[float] = None,
                       limit_to_first_n: int = None) -> [float, float]:
    
    # See https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175/28
    # for a more efficient tensor implementation
    # Apply softmax to convert logits to probabilities
    # logits: (sequence_length, vocab_size)
    
    probabilities = [torch.softmax(scores.double(), dim=-1) for scores in logits]
    
    token_probabilities = [] if prefix_probabilities is None else prefix_probabilities
    
    for i, token_probs in enumerate(probabilities):
        token_probabilities.append(token_probs[0][generated_tokens[i]])
    del probabilities
    
    if limit_to_first_n:
        token_probabilities = token_probabilities[-limit_to_first_n:]
    
    neg_log_likelihood = -torch.log(torch.clamp(torch.tensor(token_probabilities).double(), min=1e-9, max=float('inf'))) + 1e-3  # Add a small constant for numerical stability
    
    avg_neg_log_likelihood = torch.mean(neg_log_likelihood)
    del neg_log_likelihood
    
    perplexity = torch.exp(avg_neg_log_likelihood)
    del avg_neg_log_likelihood
    
    return perplexity.item(), min(token_probabilities)


def generate_unique_output_dir(base_dir):
    # Generate a unique subdirectory name based on the current date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir = os.path.join(base_dir, f"output_{timestamp}")
    return unique_dir


IGNORE_INDEX: int = -100
DEFAULT_PAD_TOKEN: str = "<pad>"
DEFAULT_EOS_TOKEN: str = "</s>"
DEFAULT_BOS_TOKEN: str = "<s>"
DEFAULT_UNK_TOKEN: str = "<unk>"


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if "attention_mask" in item:
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "labels": torch.tensor(item["labels"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long)
            }
        else:
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "labels": torch.tensor(item["labels"], dtype=torch.long)
            }


def ensure_short_training_prompts(text: str, tokenizer, max_tokens: int, base_prompt: str) -> List[str]:
    """
    For prompts that already have Bob's answers, make sure that they fit within the maximum number of tokens.
    If not, return a list of prompts where each is a continuation of the former, with only 1:Yes sentences
    kept (to try and teach the model not to keep sentences whose relevant information was already provided
    by previous sentences).
    :param text:
    :param tokenizer:
    :param max_tokens:
    :param base_prompt:
    :return:
    """
    sentence_blocks: List[str] = re.split(r'(Sentence:(?:.|\r|\n)*?###END\n\n)', text)
    sentence_blocks.pop(0)  # Remove text prior to first occurrence of "Sentence:"
    sentence_blocks = [line for line in sentence_blocks if line]
    bob_cares_sentences: List[str] = []
    bob_cares_blocks: List[str] = []
    blocks_concat = ""
    n_tokens = tokenized_length(base_prompt, tokenizer)
    prompt = base_prompt
    result: List[str] = []
    first: bool = True
    for sentence_block in sentence_blocks:
        lines = sentence_block.splitlines(keepends=False)
        if len(lines) < 6:
            raise ValueError(f'Sentence block too short: {sentence_block} from {text}')
        if re.search(r'1\s*:Yes', lines[5], re.IGNORECASE) is not None:
            bob_cares_sentences.append(lines[1])
            bob_cares_blocks.append(sentence_block)
        n_tokens += tokenized_length(sentence_block + "\n", tokenizer) + tokenized_length(lines[1] + ". ", tokenizer)
        if n_tokens > max_tokens:
            result.append(prompt + "\n```\n\n" + blocks_concat)
            prompt = base_prompt + join_strings_with_period(bob_cares_sentences)
            blocks_concat = '\n'.join(bob_cares_blocks)
            n_tokens = tokenized_length(prompt + "\n```\n\n" + blocks_concat, tokenizer)
        if first:
            first = False
            prompt = prompt + '\n' + lines[1]
        else:
            prompt = join_strings_with_period([prompt, lines[1]])
        blocks_concat = '\n'.join([blocks_concat, sentence_block])
    result.append(prompt + "\n```\n\n" + blocks_concat)
    return result


TENTATIVE_MAX_LENGTH_BEFORE_COMPLETION: int = 2048 - 130


def process_abstract(string: str, debug_files_base_name: str = '', temperature: float = 0.1, do_sample: bool = False) -> Tuple[List[Tuple[str, bool, float]], str]:  # Return important sentences, and annotated Bob responses
    re.sub(r"(?<![.]) {2,}", ' ', string)
    string = re.sub(r' {4,}', '  ', string)
    if string.strip('. \n') == '':
        logger.log(5, 'process_abstract given blank input')
        return [], ''
    if temperature is None:
        temperature = 0.1
    if debug_files_base_name is None:
        debug_files_base_name = ''
    n_tokens_input: int = tokenized_length(string, tokenizer)
    pos_sentences: List[Tuple[str, bool, float]] = []
    # Truncate ``` block to 2000 - 1st part of prompt
    # Then we can go with last processed sentence
    split = string.split('```')
    current_sentence_block: str = split[1].strip('. \n')
    annotated: str = ''
    counter: int = 0
    while True:
        keep_perc: float
        rest_of_block: str = ''
        passed: bool = False
        for keep_perc in [1.0, 0.5, 0.3, 0.2, 0.1]:
            counter += 1
            prompt: str
            if keep_perc == 1.0 and len(current_sentence_block) < 500:
                prompt = shortened_bob_prompt_base_with_example + '\n'
            else:
                prompt = split[0] + '```\n'
            if tokenized_length(prompt, tokenizer) + tokenized_length(current_sentence_block[:int(len(current_sentence_block) * keep_perc)], tokenizer) <= TENTATIVE_MAX_LENGTH_BEFORE_COMPLETION:
                truncation_index: int = int(len(current_sentence_block) * keep_perc)
                if keep_perc < 1.0:
                    while True:
                        if current_sentence_block[truncation_index - 1] == '.':
                            break
                        truncation_index -= 1
                passed_block_fraction: str = current_sentence_block[:truncation_index]
                shortened_prompt = prompt + passed_block_fraction + '.\n```\n\nSentence:'
                try:
                    pos_sentences_new, annotated_new, not_annotated_new, last_processed_sentence = \
                        extract_positive_sentences(model, tokenizer, prompt=shortened_prompt,
                                                   write_prompt_iterations_to_file='' if debug_files_base_name == ''
                                                   else f"{debug_files_base_name}_{counter}_",
                                                   temperature=temperature,
                                                   do_sample=do_sample)
                except (NoProcessedSentence, ListTooShort):
                    if len(passed_block_fraction) < 200 or LAST_SENTENCE_MARKER in passed_block_fraction:
                        passed_block_fraction = passed_block_fraction.replace(LAST_SENTENCE_MARKER, '')
                        pos_sentences_new = [(passed_block_fraction, True, 1)]
                        annotated_new = 'FAILED ANNOTATION: ' + passed_block_fraction
                        last_processed_sentence = passed_block_fraction
                    else:
                        logger.debug("Retrying because NoProcessedSentence")
                        continue
                except ValueError as e:
                    if len(passed_block_fraction) < 200 or LAST_SENTENCE_MARKER in passed_block_fraction:
                        passed_block_fraction = passed_block_fraction.replace(LAST_SENTENCE_MARKER, '')
                        pos_sentences_new = [(passed_block_fraction, True, 1)]
                        annotated_new = 'FAILED ANNOTATION: ' + passed_block_fraction
                        last_processed_sentence = passed_block_fraction
                    else:
                        raise ValueError(f'Problem with prompt {string}: {e}') from e
                pos_sentences = pos_sentences + pos_sentences_new
                annotated = annotated + '\n\n' + annotated_new
                index_end_last_sentence_in_block: int = passed_block_fraction.replace('°', '_').rfind(last_processed_sentence.replace('°', '_').strip(' .'))  # Searching in
                # passed_blocked_fraction rather than current_sentence_block on purpose, because of potential sentence repeats
                if index_end_last_sentence_in_block == -1:
                    # This could happen before because of sentence hallucination, but it should now be prevented upstream
                    raise ValueError(f'Could not find last sentence "{last_processed_sentence}" in {passed_block_fraction}')
                rest_of_block = current_sentence_block[index_end_last_sentence_in_block + len(last_processed_sentence):].lstrip(' \n \t.').rstrip(' \t\n')
                logger.debug(f'Last fully processed sentence is {last_processed_sentence}')
                if rest_of_block == '':
                    logger.debug('Block completed')
                else:
                    logger.debug(f'Rest of block is {rest_of_block[:100]}...')
                passed = True
                break  # keep_perc was small enough for us to have retrieved Bob's answer at least for the first sentence
        if not passed:
            raise ValueError(f'Could not break down {current_sentence_block}')
        all_stripped: str = rest_of_block.strip(' \n\t .')
        if all_stripped == '' or all_stripped == LAST_SENTENCE_MARKER:
            break
        #  Positive sentences will be retrieved multiple times into pos_sentences
        #  Also, they contribute substantially to the length
        #  So for now just remove them; (we could try to keep them if we didn't have to shorten the prompt too much)
        current_sentence_block = rest_of_block
    return pos_sentences, annotated


shortened_bob_prompt_base: str = """
Bob is an expert biologist analyzing sentences from a database record describing a ChIP-seq experiment. Bob needs to identify sentences that contain information about ChIP targets, cells processed, or treatments applied to those cells. This will help downstream text analysis to be performed in the future. Bob is not interested in fine technical detail, as his purpose is not to reproduce the experiments or to optimize them. Bob is also not **at all** interested in the technical aspect of the ChIP protocol. To perform his task, Bob outputs a numbered list of Yes/No answers about each sentence:
1. Is this sentence of interest to Bob?
2. Does it correspond to scientific background of the study, or to interpretation of its results?
3. Does it contain a file name with substrings (possibly abbreviated) that refer to sample-specific antibodies or their targets, cell line names, drugs, or treatment conditions?
4. Does it pertain solely to metadata?
5. Does it mention the specific antibodies used for IP, their catalogue numbers or manufacturers, or how they were raised?
6. Does it add **new** information (not already included in preceding sentences) about the cell line, tissue, or organ used for ChIP, or about the gene expression, overexpression or silencing status, or vectors the cells may contain?
7. Does it mention "interesting" cell treatments including e.g. drug treatments, application of stress or stimuli, or drugs to induce expression? Bob is not interested in regular cell culture techniques or cell preparation for ChIP.

Bob provides concise, professional, insightful, helpful, and truthful explanations for his answers.

Bob now analyzes *one by one* all the sentences in the text below.
```
"""

shortened_bob_prompt_base_with_example: str = """
Bob is an expert biologist analyzing sentences from a database record describing a ChIP-seq experiment. Bob's needs to identify sentences that contain information about ChIP targets, cells processed, or treatments applied to those cells. This will help downstream text analysis to be performed in the future. Bob is not interested in fine technical detail, as his purpose is not to reproduce the experiments or to optimize them. Bob is also not **at all** interested in the technical aspect of the ChIP protocol. To perform his task, Bob outputs a numbered list of Yes/No answers about each sentence:
1. Is this sentence of interest to Bob?
2. Does it correspond to scientific background of the study, or to interpretation of its results?
3. Does it contain a file name with substrings (possibly abbreviated) that refer to sample-specific antibodies or their targets, cell line names, drugs, or treatment conditions?
4. Does it pertain solely to metadata?
5. Does it mention the specific antibodies used for IP, their catalogue numbers or manufacturers, or how they were raised?
6. Does it add **new** information (not already included in preceding sentences) about the cell line, tissue, or organ used for ChIP, or about the gene expression, overexpression or silencing status, or vectors the cells may contain?
7. Does it mention "interesting" cell treatments including e.g. drug treatments, application of stress or stimuli, or drugs to induce expression? Bob is not interested in regular cell culture techniques or cell preparation for ChIP.

Bob provides concise, professional, insightful, helpful, and truthful explanations for his answers, as shown in the following example:

Sentence:
The second day, after 2 washes with RIPA-0.5, 1 wash with RIPA-0.3, 1 wash with RIPA-0, 2 washes with LiCl buffer (10 mM Tris-HCl, 0.25 M LiCl, 0.25% NP-40, and 0,25% NaDOC, pH7.4), and 2 washes with TE buffer, bound protein-DNA complexes were resuspended in elution buffer (10 mM Tris-HCl, 1mM EDTA, and 1% SDS, pH7.4) supplemented with 10 µg/ml RNase A for elution and RNA digestion, and incubated at 55 °C for 1 hour.
Bob's explanation:
The sentence describes protocol details of no relevance (hence 1:No) and gives no information about antibodies (hence 5:No), or cell genetic background (hence 6:No), cell treatments (hence 7:No), etc.
Bob's answer:
1:No  2:No  3:No  4:No  5:No  6:No  7:No  ###END

Bob now analyzes *one by one* all the sentences in the text below.
```
"""


#  For Bob summarization
def load_training_dataset(directory: str, tokenizer, max_tokens: int = 2048) -> List[Dict]:
    dataset: List[Dict] = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            parts = re.split(r"```", text, maxsplit=1)
            
            if len(parts) != 2:
                raise ValueError(file_path + " does not have expected ``` block")
            
            reworked_text = shortened_bob_prompt_base + parts[1]
            
            parts = re.split(r"\nSentence:\n", reworked_text, maxsplit=1)
            
            if len(parts) == 2:
                user_prompt, ai_response = parts
                full_text = user_prompt + "\nSentence:\n" + ai_response
                tokenized_text = tokenizer.encode(full_text, return_tensors="pt", truncation=True, padding="max_length", return_attention_mask=True, max_length=max_tokens).squeeze()
                
                if len(tokenized_text) < max_tokens:
                    tokenized_prompt = tokenizer.encode(user_prompt, return_tensors="pt", padding=False, add_special_tokens=False).squeeze()
                    length_without_padding = len(tokenizer.encode(full_text, return_tensors="pt", truncation=True, padding=False, return_attention_mask=False, max_length=max_tokens).squeeze())
                    labels = [IGNORE_INDEX] * len(tokenized_prompt) + tokenized_text[len(tokenized_prompt):length_without_padding].tolist()
                    labels += [IGNORE_INDEX] * (max_tokens - len(labels))
                    dataset.append({  # TODO Include attention_mask
                        "input_ids": tokenized_text.tolist(),
                        "labels": labels,
                        "group_id": file_name
                    })
                else:
                    for subset in ensure_short_training_prompts(reworked_text, tokenizer, max_tokens, shortened_bob_prompt_base):
                        parts = re.split(r"\nSentence:\n", subset, maxsplit=1)  # Split only at the first occurrence of "Sentence"
                        #  This split makes it possible for the loss function to ignore the base prompt and the overall
                        #  abstract (although the loss on the repeats of abstract sentences in the correct form after
                        #  each "Sentence: " occurrence IS taken into account)
                        user_prompt, ai_response = parts
                        full_text = user_prompt + "\nSentence:\n" + ai_response
                        tokenized_text = tokenizer.encode(full_text, return_tensors="pt", truncation=True, padding="max_length", return_attention_mask=True, max_length=max_tokens).squeeze()
                        if len(tokenized_text) > max_tokens:
                            raise ValueError("Length {len(tokenized_text)} for {full_text}")
                        tokenized_prompt = tokenizer.encode(user_prompt, return_tensors="pt", padding=False,
                                                            add_special_tokens=False).squeeze()
                        length_without_padding = len(
                            tokenizer.encode(full_text, return_tensors="pt", truncation=True, padding=False,
                                             return_attention_mask=False, max_length=max_tokens).squeeze())
                        labels = [IGNORE_INDEX] * len(tokenized_prompt) + tokenized_text[
                                                                          len(tokenized_prompt):length_without_padding].tolist()
                        labels += [IGNORE_INDEX] * (max_tokens - len(labels))
                        dataset.append({  # TODO Include attention_mask
                            "input_ids": tokenized_text.tolist(),
                            "labels": labels,
                            "group_id": file_name
                        })
                    warnings.warn(f"Warning: File '{file_name}' exceeded the maximum token limit ({max_tokens}) by {len(tokenized_text) - max_tokens}. It was split.")
            else:
                warnings.warn(f"Warning: File '{file_name}' does not contain the expected format. Skipping this file.")
    return dataset


def split_dataset(dataset: List[Dict], validation_split: float = 0.2, random_seed: int = 41) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_ids = np.array([d["group_id"] for d in dataset])
    unique_groups = np.unique(group_ids)
    force_into_training = [g for g in unique_groups if 'force_training' in g]
    available_groups = [g for g in unique_groups if 'force_training' not in g]
    np.random.seed(random_seed)
    np.random.shuffle(available_groups)
    
    n_val = int(len(unique_groups) * validation_split)
    val_groups = available_groups[:n_val]
    train_groups = available_groups[n_val:] + force_into_training
    
    logger.info(f'Train groups: {train_groups}')
    logger.info(f'Validation groups: {val_groups}')
    
    train_data: List[Dict[str, Any]] = []
    val_data: List[Dict[str, Any]] = []
    for group_id in train_groups:
        train_data.extend([d for d in dataset if d['group_id'] == group_id])
    
    for group_id in val_groups:
        val_data.extend([d for d in dataset if d['group_id'] == group_id])
    
    np.random.shuffle(train_data)
    
    return train_data, val_data


def lora_model_init(inference_mode: bool = False, lora_r: int = None, lora_dropout: float = None, lora_alpha: float = None, model0=None, do_resize_embeddings: bool = True, fp16: bool = False, int8: bool = False, do_freeze: bool = False) -> PeftModel:
    model0_was_provided: bool = model0 is not None
    if model0 is None:
        if int8:
            model0 = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map={'': torch.cuda.current_device()},
                local_files_only=hf_local_files_only
            )
        else:
            model0 = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map={'': torch.cuda.current_device()},
                local_files_only=hf_local_files_only
            )
        resize_embeddings(model0, tokenizer)
        do_resize_embeddings = False
    if do_resize_embeddings:
        resize_embeddings(model0, tokenizer)
    if len(model0.get_input_embeddings().weight.data) != 32001:
        raise ValueError(f'Wrong input embeddings for model0: {len(model0.get_input_embeddings().weight.data)}')
    if lora_r is None:
        lora_r = 16
    lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    if lora_dropout is None:
        lora_dropout = 0.3
    if lora_alpha is None:
        lora_alpha = 1.0
    if do_freeze or not model0_was_provided:
        for param in model0.parameters():
            param.requires_grad = False
    
    if not model0_was_provided:
        if fp16:
            model0 = prepare_model_for_float16_training(model0)
        elif int8:
            model0 = prepare_model_for_int8_training(model0)
            
    if len(model0.get_input_embeddings().weight.data) != 32001:
        raise ValueError(f'Wrong input embeddings for model0: {len(model0.get_input_embeddings().weight.data)}')
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=inference_mode, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=lora_target_modules, bias="none"
    )
    model = get_peft_model(model0, peft_config)
    model.config.use_cache = False
    model.config.pad_token_id = model0.config.pad_token_id
    model.config.eos_token_id = model0.config.eos_token_id
    model.config.bos_token_id = model0.config.bos_token_id
    
    if len(model.get_input_embeddings().weight.data) != 32001:
        raise ValueError(f'Wrong input embeddings for instantiated LoRa model: {len(model.get_input_embeddings().weight.data)}')
    
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    model.print_trainable_parameters()
    
    return model


# Copied and modified from PEFT other.py
def prepare_model_for_float16_training(
    model, output_embedding_layer_name: str = "lm_head", use_gradient_checkpointing: bool = True, layer_norm_names: List[str] = None
):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32
    
    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    if layer_norm_names is None:
        layer_norm_names = ["layer_norm"]   # NB: this was "layernorm" (no underscore)
    loaded_in_16bit = True
    
    param: torch.nn.parameter.Parameter
    
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        
        if loaded_in_16bit:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
                param.data = param.data.to(torch.float32)
    
    if loaded_in_16bit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    
    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype
        
        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is cast
            in fp32
            """
            
            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)
            
            #  Added to fix AttributeError: 'CastOutputToFloat' object has no attribute 'weight'
            #  caused by resize_token_embeddings --> new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            #                           --> old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            #  When you create an instance of the CastOutputToFloat class, you pass the output_embedding_layer as an
            #  argument. The Sequential container stores this layer as its first sub-module. Thus, when you access
            #  self[0] inside the CastOutputToFloat class, you are accessing the original output_embedding_layer that
            #  you provided when creating the instance.
            @property
            def weight(self):
                return self[0].weight
        
        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))
    
    return model


def resize_embeddings(model, tokenizer: PreTrainedTokenizer):
    if len(model.get_input_embeddings().weight.data) == 32001:
        raise ValueError('Embeddings probably already resized')
    # Update the model's embeddings to account for the new padding token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # The following is copied from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    
    input_embeddings_avg = input_embeddings[:-num_added_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_added_tokens].mean(dim=0, keepdim=True)
    
    input_embeddings[-num_added_tokens:] = input_embeddings_avg
    output_embeddings[-num_added_tokens:] = output_embeddings_avg


tokenizer: Optional[PreTrainedTokenizer] = LlamaTokenizer.from_pretrained(model_name, local_files_only=hf_local_files_only)
# Check if the tokenizer already has a padding token
num_added_tokens: int = 0
if tokenizer.pad_token is None:
    # Define a new padding token (you can choose any unused token string)
    new_pad_token = DEFAULT_PAD_TOKEN
    
    # Add the new padding token to the tokenizer's vocabulary
    num_added_tokens = tokenizer.add_special_tokens({'pad_token': new_pad_token})
    
    # Check if the padding token was successfully added
    if num_added_tokens > 0:
        pass
    else:
        tokenizer = None
        raise ValueError("Failed to add padding token to the tokenizer.")
    if tokenizer.pad_token_id != 32000:
        tokenizer = None
        raise ValueError("Probably failed to add padding token to the tokenizer.")

initial_stopping_criteria_tuple: Tuple[str, int, int, int, str] = \
    (LAST_SENTENCE_MARKER, len(tokenizer.encode(LAST_SENTENCE_MARKER)) + 1, 1, 0, '###END')


class SaveFirstEpochCallback(TrainerCallback):
    def __init__(self, custom_params):
        super().__init__()
        self.custom_params = custom_params
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Check if the current epoch is the first epoch (epoch numbering starts from 0)
        logger.info(f'Epoch {state.epoch} end')
        if True or state.epoch == 1.0:
            # Save the model checkpoint
            checkpoint_folder = f"{args.output_dir}/checkpoint-epoch-{str(state.epoch)}"
            kwargs["model"].save_pretrained(checkpoint_folder)
            # kwargs["tokenizer"].save_pretrained(checkpoint_folder)
            
            # Issue in alpaca-lora repo suggests removing the pytorch_model.bin output file https://github.com/tloen/alpaca-lora/issues/319
            
            # Save the custom parameters to a JSON file
            json_file = f"{checkpoint_folder}/custom_params.json"
            with open(json_file, "w") as f:
                json.dump(self.custom_params, f, indent=4)


class GradientLoggingCallback(TrainerCallback):
    # actually need to override training_step in trainer.py to get the gradient;
    # gradient isn't there when on_step_end callback is made
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Compute the average gradient magnitude
        grad_magnitude = 0.0
        num_params = 0
        for name, param in kwargs['model'].named_parameters():
            if param.grad is not None:
                grad_magnitude += torch.norm(param.grad).item()
                num_params += 1
        
        if num_params == 0:
            logger.debug("No named parameters with gradient in model")
            pass
        else:
            logger.info(f"Step {state.global_step}: Average Gradient Magnitude = {grad_magnitude / num_params}")


def trainer(dataset: list[dict[str, Any]] = None,
            learning_rate: float = None,
            weight_decay: float = None,
            num_epochs: int = None,
            model=None,
            model0=None,  # Underlying model for LoRa
            early_stop_patience: int = None,
            early_stop_rel_improvement_threshold: float = 0.2,
            output_dir_base_name: str = None,
            fp16: bool = None,
            micro_batch_size: int = 1,
            parameters_to_save: Dict[str, Any] = None,
            validation_split: float = 0.2
            ) -> Trainer:
    if parameters_to_save is None:
        parameters_to_save = {}
    if model is not None and model0 is not None:
        raise ValueError('Passed model0 is not None but it would not be used because model is also not None')
    if learning_rate is None:
        learning_rate = 4e-4
    if weight_decay is None:
        weight_decay = 0
    if num_epochs is None:
        num_epochs = 4
    if fp16 is None:
        fp16 = False
    if output_dir_base_name is None:
        output_dir_base_name = generate_unique_output_dir("./training")
    output_dir = output_dir_base_name + '_out'
    log_dir = output_dir_base_name + '_log'
    
    if micro_batch_size is None:
        micro_batch_size = 1
    
    if dataset is None:
        logger.warning("Automatically making dataset from summarization_training")
        dataset: list[dict[str, Any]] = load_training_dataset("bob_training_samples", tokenizer)
        
    train_data, val_data = split_dataset(dataset, validation_split=validation_split)
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    
    early_stop = RelativeImprovementEarlyStoppingCallback(
        early_stopping_patience=early_stop_patience,
        threshold=early_stop_rel_improvement_threshold,
        metric_name='eval_loss' if len(val_data) > 0 else 'loss'
    )
    
    save_first_epoch_callback = SaveFirstEpochCallback(custom_params={**parameters_to_save, **{
                                                                       "weight_decay": weight_decay,
                                                                      "learning_rate": learning_rate,
                                                                      "micro_batch_size": micro_batch_size,
                                                                      "per_device_train_batch_size": micro_batch_size,
                                                                      "gradient_accumulation_steps": micro_batch_size
                                                                      }})
    
    trainer = Trainer(
        model=lora_model_init(model0=model0) if model is None else model,
        model_init=None,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stop, save_first_epoch_callback, GradientLoggingCallback()],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=micro_batch_size * 2 if micro_batch_size != 1 else 1,
            per_device_eval_batch_size=1,  # Necessary under some circumstances to avoid GPU out of memory errors
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            logging_steps=3,
            optim="adamw_torch",     # optim="adamw_bnb_8bit" behaves badly (numerical instability??)
                                     # See https://github.com/huggingface/transformers/issues/14819
                                     # https://huggingface.co/docs/transformers/v4.23.1/en/perf_train_gpu_one
            evaluation_strategy="steps" if len(val_data) > 0 else "no",  # evaluate every `eval_steps` steps ;
            save_strategy="steps",
            eval_steps=len(train_data)//10 if len(val_data) > 0 else None,
            save_steps=len(train_data)//3,
            output_dir=output_dir,
            logging_dir=log_dir,
            load_best_model_at_end=True if len(val_data) > 0 else False,
            group_by_length=True,  # Irrelevant for us since we're not batching
            report_to=None,
            run_name=None,
        ),
    )
    return trainer


class RelativeImprovementEarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold: float, metric_name: str = 'eval_loss', early_stopping_patience: int = 1):
        super().__init__()
        self.threshold = threshold
        self.metric_name = metric_name
        if early_stopping_patience is None:
            early_stopping_patience = 1
        self.early_stopping_patience = early_stopping_patience
        self.last_metric = None
        self.best_metric = None  # Currently unused
        self.patience_counter = 0
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = state.log_history[-1]
        try:
            current_metric = metrics.get(self.metric_name)
        except KeyError as ke:
            raise ValueError(f'Could not find key {self.metric_name} in {metrics}')
        if self.last_metric is None:
            self.last_metric = current_metric
            return
        
        # Calculate relative improvement
        relative_improvement = (current_metric - self.last_metric) / abs(self.last_metric)
        
        if not args.greater_is_better:
            relative_improvement = -relative_improvement
        
        operator = np.greater if args.greater_is_better else np.less
        
        if self.best_metric is None or operator(current_metric, self.best_metric):
            self.best_metric = current_metric
        
        self.last_metric = current_metric
        
        # Update the last metric and reset patience counter if there is sufficient improvement
        if relative_improvement >= self.threshold:
            self.patience_counter = 0  # Reset patience counter
        else:
            self.patience_counter += 1  # Increment patience counter
            logger.log(5, f"Patience: {self.patience_counter} out of {self.early_stopping_patience}")
        
        # Check if the relative improvement is below the threshold for enough steps
        if self.patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
        
        return control


def train_barb(model0=None, model=None, output_dir_base: str = None, fp16: bool = None, trainer_fp16: bool = None, lora_r: int = 16, do_resize_embeddings: bool = True, int8=False,
               lora_dropout=0.3, lora_alpha=1.0, learning_rate=None, weight_decay: float = 0.02, micro_batch_size: int = 1, validation_split: float = 0.2) -> Trainer:
    if output_dir_base is None:
        output_dir_base = 'barb'
    if model is None:
        model = lora_model_init(lora_r=lora_r, lora_dropout=lora_dropout, lora_alpha=lora_alpha, model0=model0, do_resize_embeddings=do_resize_embeddings, fp16=fp16, int8=int8)
    parameters_to_save: Dict[str, Any] = {"fp16": fp16, "trainer_fp16": trainer_fp16, "int8": int8, "lora_r": lora_r, "lora_dropout": lora_dropout, "lora_alpha": lora_alpha}
    return trainer(parameters_to_save=parameters_to_save, model=model, fp16=trainer_fp16, learning_rate=learning_rate, micro_batch_size=micro_batch_size, weight_decay=weight_decay, early_stop_rel_improvement_threshold=0.05, early_stop_patience=20, dataset=flattened_barb, output_dir_base_name=output_dir_base, validation_split=validation_split)
