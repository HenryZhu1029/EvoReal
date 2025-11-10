from typing import Optional
import logging
import json
import random
from datetime import datetime
import subprocess
import numpy as np
import os
import shutil
from pathlib import Path
from omegaconf import DictConfig
from utils.utils import *

class CVRPDataEvo:
    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str,
        generator_llm,
        debug_mode: bool,
        reflector_llm: Optional[object] = None,
    ) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.model = cfg.model

        # Running Logger
        self.iteration = 0
        self.function_evals = 0
        self.max_code_runs = 0

        # population + elite + best module initialization
        self.population_cvrp = []
        self.elitist_cvrp = None
        self.best_cvrp_module = None

        self.best_cvrp_gap= None
        self.best_cvrp_iterID= 0
        # path 
        if hasattr(cfg, "log_prefix") and self.cfg.log_prefix:
            self.log_prefix = self.cfg.log_prefix
        else:
            self.log_prefix = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_type{self.cfg.module_to_evo.upper()}" + f"_{self.model}"
        self.log_save_path = os.path.join(self.root_dir, "llm_log", f"{self.log_prefix}_llm_log", "gpt")
        self.gpt_save_path = os.path.join(self.root_dir, "llm_log", f"{self.log_prefix}_llm_log")
        self.checkpoint_folder = os.path.join(self.root_dir,"llm_log", f"{self.log_prefix}_llm_log", "checkpoints")
        os.makedirs(self.log_save_path, exist_ok=True)
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        
        # src_path = os.path.join(self.root_dir, "llm_log", "generate_instance_cvrp.py")
        # dst_path = os.path.join(self.gpt_save_path, "generate_instance_cvrp.py")
        # shutil.copyfile(src_path, dst_path)

        # Hyperparameter setting
        self.mutation_rate = cfg.mutation_rate


        # prompt and reflection
        self.long_term_reflection_str_cvrp = ""

        # Early Stopping setup
        self.early_stopping_cvrp = False
        self.toleration = self.cfg.toleration
        self.no_improvement_counter_cvrp = 0

        # iteration setups
        self.max_iterations = self.cfg.max_iter
        self.DEBUG_MODE = debug_mode
        if self.DEBUG_MODE:
            self.max_iterations = 1
        self._module_to_evo = self.cfg.module_to_evo
        # CUDA setup
        self.cuda_device_num = self.cfg.cuda_device_num

        
    def init_prompt(self) -> None:
        
        self.problem = self.cfg.problem.problem_name
      
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.module_cvrp_path = f"{self.root_dir}/llm_log/{self.log_prefix}_llm_log/coord_generator.py"

        # Loading all text prompts
        # Problem-specific prompt components
        problem_prompt_path = f'{self.prompt_dir}'
        
        self.func_signature_cvrp = file_to_string(f'{problem_prompt_path}/func_signature.txt')

        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        self.problem_desc = file_to_string(f'{problem_prompt_path}/problem_desc.txt')
        

        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)

               
        # Common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(f'{self.prompt_dir}/user_reflector_st.txt') # shrot-term reflection
        self.user_reflector_lt_prompt = file_to_string(f'{self.prompt_dir}/user_reflector_lt.txt') # long-term reflection
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/crossover.txt')
        self.mutataion_prompt = file_to_string(f'{self.prompt_dir}/mutation.txt')

        ### Heuristic Design Guidance
        self.design_guidance = file_to_string(f'{self.prompt_dir}/design_guidance.txt')

        
        ### Seed Functions
        self.seed_generator_cvrp = file_to_string(f'{self.prompt_dir}/seed_generator.txt')

        self.cvrp_func_name = "generate_cvrp_syn"


        self.seed_cvrp_prompt = file_to_string(f'{self.prompt_dir}/seed.txt').format(
            external_guide = self.design_guidance,
            seed_func=self.seed_generator_cvrp,
            func_name=self.cvrp_func_name,
        )

        # Flag to print prompts
        self.print_system_generator_prompt = False
        self.print_crossover_prompt = False # Print crossover prompt for the first iteration
        self.print_mutate_prompt = False # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True # Print short-term reflection prompt for the first iteration
        self.print_long_term_reflection_prompt = True # Print long-term reflection prompt for the first iteration
        self.print_gen_population_prompt = False
        

    
    def gen_init_population(self, module_type: str, id_prefix: int, batch_size:int) -> list:
        """Generate initial population for a specific module type."""
        logging.info(f"Initializing initial population for module {module_type}...")

        # Prompt components per module type
        func_name = getattr(self, f"{module_type}_func_name")
        func_signature = getattr(self, f"func_signature_{module_type}")
        seed_prompt = getattr(self, f"seed_{module_type}_prompt")
        
        system = self.system_generator_prompt
        user_prompt = self.cfg.prompts.population_generator_prompt.format(
            # func_name=func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            func_signature=func_signature,
            seed=seed_prompt
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]

        responses = self.generator_llm.multi_chat_completion(
            [messages], 
            temperature=self.cfg.temperature + 0.1,
            n=int(batch_size),
            concurrent=False
        )

        population = [
            self.response_to_individual(response, response_id=id_prefix+response_id, module_type=module_type)
            for response_id, response in enumerate(responses)
        ]

        if self.print_system_generator_prompt:
            logging.info("Population Generator Prompt:\nSystem Prompt:\n" + system)
            self.print_system_generator_prompt = False
        if self.print_gen_population_prompt:
            logging.info("Population Generator Prompt:\nUser Prompt:\n" + user_prompt)
            if self.iteration>0:
                self.print_gen_population_prompt = False

        return population
    
    def add_seed_to_population(self, module_type: str=None):
 
        seed_code = getattr(self, f"seed_generator_{module_type}")
        func_name = getattr(self, f"{module_type}_func_name") 
        file_name = f"llm_log/{self.log_prefix}_llm_log/gpt/iter{self.iteration}_module{module_type}_resp0.txt"
        with open(file_name, 'w') as f:
            f.write(seed_code + "\n")

        stdout_path = f"llm_log/{self.log_prefix}_llm_log/gpt/iter{self.iteration}_module{module_type}_resp0_stdout.txt"

        individual = {
            "func_name": func_name,
            "stdout_filepath": stdout_path,
            "code_path": file_name,
            "code": seed_code,
            "response_id": 0,
            "module_type": module_type,
        }

        try:
            aug_gap, non_aug_gap = self._run_code(
                individual=individual,
                response_id=0,
                module_type=module_type,
            )
        except Exception as e:
            logging.error(f"[Seed Run] {module_type} runtime error: {e}")
            self.mark_invalid_individual(individual, f"Seed runtime error: {e}")
            aug_gap = non_aug_gap = None
        print(f"[DEBUG-Seed] log_prefix: {self.log_prefix}")
        print(f"[DEBUG-Seed] checkpoint_folder: {self.checkpoint_folder}")
        print(f"[DEBUG-Seed] code_path: {individual['code_path']}")
        print(f"[DEBUG-Seed] stdout_path: {individual['stdout_filepath']}")
        print(f"[DEBUG-Seed] module_type: {module_type}")

        if aug_gap is None or not isinstance(aug_gap, float):
            individual.update(
                {
                    "exec_success": False,
                    "aug_gap": float("inf"),
                    "non_aug_gap": float("inf"),
                }
            )
        else:
            individual.update(
                {
                    "exec_success": True,
                    "aug_gap": aug_gap,
                    "non_aug_gap": non_aug_gap,
                }
            )
            logging.info(f"[Seed Run] {module_type}: aug_gap={aug_gap:.4f}, non_aug_gap={non_aug_gap:.4f}")

        getattr(self, f"population_{module_type}").append(individual)
   
    
    def response_to_individual(self, response: str, response_id: int, module_type: str=None) -> dict:
        # os.makedirs("llm_log", exist_ok=True)
        # Write response to file
        file_name = f"llm_log/{self.log_prefix}_llm_log/gpt/iter{self.iteration}_module{module_type}_resp{response_id}.txt" 
        func_name = getattr(self, f"{module_type}_func_name")
        
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)

        std_out_filepath = f"llm_log/{self.log_prefix}_llm_log/gpt/iter{self.iteration}_module{module_type}_resp{response_id}_stdout.txt"
        if code is None:
            logging.warning(f"[Code Extraction Failed] Module {module_type}, Response {response_id}")
            return {
                "func_name": func_name,
                "stdout_filepath": None,
                "code_path": file_name,
                "code": None,
                "response_id": response_id,
                "module_type": module_type
            }
        else:
            individual = {
                "func_name": func_name,
                "stdout_filepath": std_out_filepath,
                "code": code,
                "code_path":file_name,
                "response_id": response_id,
                "module_type": module_type
            }
            return individual

        
    def _run_code(self, individual: dict, response_id: int, module_type: str = None):
        logging.debug(f"Iteration {self.iteration}: Running Code {response_id} (internal eval)")
        code_dump_path = getattr(self, f"module_{module_type}_path")

    
        # check if code contains pseudo-codes (non ASCII characters)
        if not all(ord(c) < 128 for c in individual["code"]):
            self.mark_invalid_individual(individual, "Contains non-ASCII (possibly corrupted or garbage code).")
            return None, None
        if individual["code"] is None or not isinstance(individual["code"], str):
            self.mark_invalid_individual(individual, "No code extracted or invalid format.")
            return None, None
    
        # Write generated code to file
        with open(code_dump_path, 'w') as f:
            f.write(individual["code"] + '\n')
    
        stdout_path = individual["stdout_filepath"]
    
        # Run subprocess and capture + print output in real-time
        with open(stdout_path, 'w') as f:
            try:
                process = subprocess.Popen(
                    ["python", "-u", "POMO/get_fitness_score.py",
                     "--checkpoint_folder", self.checkpoint_folder,
                     "--idx_iteration", str(self.iteration),
                     "--idx_response_id", str(response_id),
                     "--cuda_device_num", str(self.cuda_device_num),
                     "--module_type", module_type,
                    "--log_prefix", self.log_prefix
                    ],
                     
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
    
                all_output = ""
                start_time = time.time()
                timeout_seconds = 3600
    
                while True:
                    line = process.stdout.readline()
                    if line == '' and process.poll() is not None:
                        break
                    if line:
                        print(line, end='')    
                        f.write(line)           
                        all_output += line
    

                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        process.kill()
                        f.write("Traceback: Timeout after 25 minutes.\n")
                        logging.error(f"Timeout: Process {response_id} killed after 25 minutes.")
                        return None, None
    
                if not all_output.strip():
                    logging.warning("[Subprocess] Empty stdout received.")
    
                score_output = extract_json_from_stdout(all_output)
                if score_output is None:
                    f.write("Traceback: Failed to extract valid JSON result from subprocess stdout")
                    return None, None
    
                return score_output.get("aug_gap"), score_output.get("non_aug_gap")
    
            except Exception as e:
                f.write("Traceback: " + str(e) + '\n')
                logging.error(f"Execution failed for response {response_id}: {e}")
                return None, None

    
    def evaluate_population(self, population: list[dict], module_type: str=None) -> list[dict]:
        """
        Evaluate each individual by executing its code and computing aug/non-aug gap
        using average of LB & UB from self.train_set_label.
        """
        expected_func_name = getattr(self, f"{module_type}_func_name")
        for individual in population:
            self.function_evals += 1
            response_id = individual['response_id']

            if individual.get("code") is None:
                self.mark_invalid_individual(individual, "Invalid response (no code).")
                continue
            if individual.get("func_name") != expected_func_name:
                self.mark_invalid_individual(individual, f"Function name mismatch: expected {expected_func_name}, got {individual.get('func_name')}")
                continue

            logging.info(f"[Evaluation] Iteration {self.iteration}: Running Code {response_id}, Module Type:{module_type}")

            # Run TRAIN + EVAL
            try:
                aug_gap, non_aug_gap = self._run_code(individual, response_id, module_type=module_type)
            except Exception as e:
                logging.error(f"Runtime error during execution of response {response_id}: {e}")
                self.mark_invalid_individual(individual, f"Runtime error: {e}")
                continue

            if aug_gap is None or not isinstance(aug_gap, float):
                self.mark_invalid_individual(individual, "aug gap is None (execution failed)")
                logging.info("***FAILED to get valid SCORE!!!***")
                self.max_code_runs += 1
                continue
            else:
                individual["exec_success"] = True
                self.max_code_runs += 1
                logging.info("***INDIVIDUAL IS MARKED AS SUCESSFUL!!!***")
                
            individual["aug_gap"] = aug_gap
            individual["non_aug_gap"] = non_aug_gap


        return population


    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> None:
        """
        In-place mark an individual as invalid and clear any performance metrics.
        """
        individual["exec_success"] = False
        individual["aug_gap"] = float("inf")
        individual["non_aug_gap"] = float("inf")
        individual["traceback_msg"] = traceback_msg


        
    def crossover(self, id_prefix: int, module_type: str, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        """
        Perform crossover using short-term reflections. 
        Combine better and worse individuals based on pairwise LLM feedback.
        """
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        messages_lst = []

        # Dynamically obtain the prompt meta-information of this module type
        func_signature = getattr(self, f"func_signature_{module_type}")
        func_name = getattr(self, f"{module_type}_func_name", f"generate_{module_type.lower()}_type")

        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            system = self.system_generator_prompt

            user = self.crossover_prompt.format(
                func_signature=func_signature,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                # reflection="", # only used for ablation study(remove short-term reflection only)
                # func_name=func_name,
                problem_desc=self.problem_desc,
                func_desc=self.func_desc
            )

            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)

            if self.print_crossover_prompt:
                logging.info("Crossover Prompt:\nSystem Prompt:\n" + system + "\nUser Prompt:\n" + user)
                if self.iteration>0:
                    self.print_crossover_prompt = False

        responses = self.generator_llm.multi_chat_completion(messages_lst)

        crossed_population = [
            self.response_to_individual(response, id_prefix * 100 + response_id, module_type=module_type)
            for response_id, response in enumerate(responses)
        ]

        return crossed_population

    
    
    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        module_type = ind1["module_type"]
        if ind1["aug_gap"] == ind2["aug_gap"]:
            print(ind1["code"], ind2["code"])
            return None, None, None

        # Determine better vs worse
        if ind1["aug_gap"] < ind2["aug_gap"]:
            better_ind, worse_ind = ind1, ind2
        else:
            better_ind, worse_ind = ind2, ind1

        # Filtered code for prompt
        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])

        # Get correct function name and signature
        func_name = getattr(self, f"{module_type}_func_name")
        func_signature = getattr(self, f"func_signature_{module_type}")

        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            # func_name=func_name,
            func_signature=func_signature,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_code,
            better_code=better_code,
        )

        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        if self.print_short_term_reflection_prompt:
            logging.info("Short-term Reflection Prompt:\nSystem:\n" + system + "\nUser:\n" + user)
            self.print_short_term_reflection_prompt = False

        return message, worse_code, better_code


    
    def short_term_reflection(self, population: list[dict], module_type: str = None) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []

        # Filter executable individuals
        population = [ind for ind in population if ind.get("exec_success", False) and ind.get("code") is not None]

        # Handle odd-size population
        use_extra_pair = False
        extra_individual = None
        elitist = getattr(self, f"elitist_{module_type}")

        if len(population) % 2 != 0:
            extra_individual = population[-1]
            population = population[:-1]
            if elitist and elitist.get("exec_success", False):
                use_extra_pair = True
            else:
                logging.warning("Elitist unavailable or invalid. Dropping last individual for pairing.")

        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i + 1]

            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            if messages is None:
                continue  # Skip if reflection couldn't be generated
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)

        if use_extra_pair:
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(extra_individual, elitist)
            if messages is not None:
                messages_lst.append(messages)
                worse_code_lst.append(worse_code)
                better_code_lst.append(better_code)

        if not messages_lst:
            return [], [], []

        response_lst = self.reflector_llm.multi_chat_completion(messages_lst)
        return response_lst, worse_code_lst, better_code_lst

    
    def long_term_reflection(self, short_term_reflections: list[str], module_type: str) -> None:
        """
        Long-term reflection for a given module type before mutation.
        """
        system = self.system_reflector_prompt
        prior_reflection = getattr(self, f"long_term_reflection_str_{module_type}", "")
        
        user = self.user_reflector_lt_prompt.format(
            problem_desc=self.problem_desc,
            prior_reflection=prior_reflection,
            new_reflection="\n".join(short_term_reflections),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        if self.print_long_term_reflection_prompt:
            logging.info(f"[{module_type}] Long-term Reflection Prompt:\nSystem Prompt:\n" + system + "\nUser Prompt:\n" + user)
            self.print_long_term_reflection_prompt = False

        # Generate and store long-term reflection
        reflection = self.reflector_llm.multi_chat_completion([messages])[0]
        setattr(self, f"long_term_reflection_str_{module_type}", reflection)

        # Write reflections to disk
        log_dir = os.path.join(self.root_dir,"llm_log", f"{self.log_prefix}_llm_log", "reflections")
        os.makedirs(log_dir, exist_ok=True)

        short_term_file = os.path.join(log_dir, f"iter{self.iteration}_{module_type}_short_term.txt")
        with open(short_term_file, 'w') as f:
            f.writelines("\n".join(short_term_reflections) + '\n')

        long_term_file = os.path.join(log_dir, f"iter{self.iteration}_{module_type}_long_term.txt")
        with open(long_term_file, 'w') as f:
            f.write(reflection + '\n')

    
    def mutate(self, id_prefix: int, module_type: str = None) -> list[dict]:
        """
        Elitist-based mutation. We mutate the best individual (elitist) to generate new variants.
        """
        system = self.system_generator_prompt

        elitist = getattr(self, f"elitist_{module_type}")        
        func_signature = getattr(self, f"func_signature_{module_type}")
        long_term_reflection = getattr(self, f"long_term_reflection_str_{module_type}", "")
        func_name = getattr(self, f"{module_type}_func_name")

        if elitist is None or elitist.get("code") is None:
            logging.warning(f"[Mutation-{module_type}] Elitist is missing or invalid. Skipping mutation.")
            return []

        user = self.mutataion_prompt.format(
            reflection=long_term_reflection,
            func_signature=func_signature,
            elitist_code=filter_code(elitist["code"]),
            # func_name=func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        if self.print_mutate_prompt:
            logging.info(f"[{module_type}] Mutation Prompt:\nSystem Prompt:\n" + system + "\nUser Prompt:\n" + user)
            self.print_mutate_prompt = False

        responses = self.generator_llm.multi_chat_completion(
            [messages],
            temperature=self.cfg.temperature + 0.2,
            n=int(self.cfg.pop_size * self.mutation_rate),
            concurrent=False
        )

        mutated_population = [
            self.response_to_individual(response, 1000 * id_prefix + response_id, module_type=module_type)
            for response_id, response in enumerate(responses)
        ]

        return mutated_population



    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals

        population = [individual for individual in population if individual["exec_success"]]
        
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < min(self.cfg.pop_size,len(population)):
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["aug_gap"] != parents[1]["aug_gap"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """

        population = [individual for individual in population if individual["exec_success"]]
        
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["aug_gap"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < min(self.cfg.pop_size,len(population)):
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["aug_gap"] != parents[1]["aug_gap"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    def update_elitist(self, module_type: str) -> None:
        population = getattr(self, f"population_{module_type}")
        elitist = getattr(self, f"elitist_{module_type}")

        # Get the aug gap list
        aug_gaps = [ind["aug_gap"] for ind in population if ind.get("exec_success", False)]
        best_gap = min(aug_gaps)
        # Only retain the individuals who have successfully executed
        valid_population = [ind for ind in population if ind.get("exec_success", False)]

        # Find the individual with the smallest aug_gap
        best_individual = min(valid_population, key=lambda ind: ind["aug_gap"])
        best_gap = best_individual["aug_gap"]

        # update current elite
        if elitist is None or best_gap < elitist["aug_gap"]:
            setattr(self, f"elitist_{module_type}", best_individual)
            logging.info(f"[{module_type}] Iter {self.iteration}: Elitist updated, aug_gap = {best_gap:.4f}")

        # update best individual
        best_gap_attr = getattr(self, f"best_{module_type}_gap")
        if best_gap_attr is None or best_gap < best_gap_attr:
            setattr(self, f"best_{module_type}_gap", best_gap)
            setattr(self, f"best_{module_type}_module", best_individual["code"])
            setattr(self, f"best_{module_type}_iterID", self.iteration)
            logging.info(f"[{module_type}] Iter {self.iteration}: Best gap updated to {best_gap:.4f}")
            setattr(self, f"no_improvement_counter_{module_type}", 0)
        else:
            count = getattr(self, f"no_improvement_counter_{module_type}", 0) + 1
            setattr(self, f"no_improvement_counter_{module_type}", count)
            logging.info(f"[{module_type}] Iter {self.iteration}: No improvement count = {count}")
            if count >= self.toleration:
                setattr(self, f"early_stopping_{module_type}", True)
                logging.info(f"[{module_type}] Early stopping triggered (no improvement)")

       
    def evolve(self):
        assert self.cfg.init_pop_size<=30, "maximum number of initial population of each type should not be larger than 30!"
        self.init_prompt()
        self.module_list = ["cvrp"]

                
        # Generate Initial Population
        # for prefix,module_type in enumerate(self.module_list.copy()):
        module_type = self._module_to_evo
        #if module_type == self.module_list[0]:
          #self.num_total_training_epochs = self.num_total_training_epochs_list[0]
        #elif module_type == self.module_list[1]:
          #self.num_total_training_epochs = self.num_total_training_epochs_list[1]
        #else:
         # self.num_total_training_epochs = self.num_total_training_epochs_list[2]
          
          
        if module_type in self.module_list:
            logging.info(f"---Generate initial {module_type} population---")
            self.add_seed_to_population(module_type=module_type)
            logging.info(f"---Seed of {module_type} type has been added.---")
            pop = getattr(self, f"population_{module_type}")
            id_prefix = 1
            num_batches = self.cfg.num_batches
            batch_size = self.cfg.init_pop_size // num_batches
            for batch_id in range(num_batches):
                logging.info(f"Generating batch {batch_id+1}/{num_batches} for module {module_type}")
                init_pop = self.gen_init_population(module_type=module_type,id_prefix=id_prefix,batch_size=batch_size)    
                pop.extend(self.evaluate_population(init_pop, module_type))
                id_prefix+=batch_size
            self.update_elitist(module_type)
        else:
            print(f"Module Type{module_type} is invalid! Choose from{self.module_list}!")
            raise TypeError
                         
        while self.iteration < self.max_iterations and self.max_code_runs < self.cfg.max_code_runs:
            logging.info(f"========== Iteration {self.iteration} ==========")
            

            self._module_to_evo == self.module_list[0]
            id_prefix = 1

            logging.info(f"---Begin Evolution for Module {module_type} at iteration {self.iteration}: ---")

            pop = getattr(self, f"population_{module_type}")
            elitist = getattr(self, f"elitist_{module_type}")
            
            # Check for at least two valid individuals
            valid_pop = [ind for ind in pop if ind.get("exec_success", False)]
            if len(valid_pop) < 2:
                print(f"[{module_type}] Not enough valid individuals to perform crossover (found {len(valid_pop)}).")
                raise RuntimeError()

            
            short_term_reflection_tuple = self.short_term_reflection(pop, module_type)
            crossed_population = self.crossover(id_prefix=id_prefix,module_type=module_type,short_term_reflection_tuple=short_term_reflection_tuple)
            pop.extend(self.evaluate_population(crossed_population, module_type))

            self.update_elitist(module_type)

            if getattr(self, f"early_stopping_{module_type}", False):
                logging.info(f"[{module_type}] Early stopping at iteration {self.iteration}")
                #self.module_list.remove(module_type)
                final_elitist = getattr(self,f"elitist_{module_type}")
                return final_elitist
                
            # comment only for ablation study
            self.long_term_reflection([response for response in short_term_reflection_tuple[0]],module_type=module_type)
            mutated_population = self.mutate(module_type=module_type,id_prefix=id_prefix)
            pop.extend(self.evaluate_population(mutated_population, module_type))

            if elitist is None or any(ind["response_id"] == elitist["response_id"] for ind in pop):
                population_to_select = pop
            else:
                population_to_select = [elitist] + pop
            selected_population = self.rank_select(population_to_select)

            if selected_population is None:
                raise RuntimeError(f"Rank-based selection failed for module {module_type}")


            self.population_cvrp = selected_population if elitist is None or any(ind["response_id"] == elitist["response_id"] for ind in selected_population) else [elitist] + selected_population
 
 
                
            #update iter
            self.iteration += 1
            logging.info(f"Proceeding to Iteration {self.iteration}")

        logging.info("=== Termination Status ===")
        if self.max_code_runs >= self.cfg.max_code_runs:
            logging.info(f"Stopped early: reached max_code_runs = {self.max_code_runs}")
        elif self.iteration >= self.max_iterations:
            logging.info(f"Stopped: reached max_iterations = {self.max_iterations}")

        final_elitist = getattr(self,f"elitist_{module_type}")
        return final_elitist






