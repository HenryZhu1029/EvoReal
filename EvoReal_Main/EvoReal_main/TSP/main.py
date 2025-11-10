import hydra
import logging
import os
import json
import sys
import torch
from pathlib import Path
from utils.utils import init_client
from tspdata_evo_pomo import TSPDataEvo as POMO_EVO
from tspdata_evo_lehd import TSPDataEvo as LEHD_EVO


print("sys.getdefaultencoding():", sys.getdefaultencoding())
os.environ["HYDRA_FULL_ERROR"] = "1"
logging.basicConfig(level=logging.INFO)
# ROOT_DIR = os.getcwd()
DEBUG_MODE = False

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    """
    Entry point for LLM-driven TSP data evolution:
    1. Initialize LLM client and evolutor
    2. Run multi-round heuristic search with prompt tuning
    3. Save the best evolved code
    """
    # Path Config
    ROOT_DIR = Path.cwd()
    best_code_dir = ROOT_DIR / "best_code"
    best_code_dir.mkdir(parents=True, exist_ok=True)
    hydra_dir = ROOT_DIR / "hydra_outputs"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"[Hydra Workspace] Working Directory: {ROOT_DIR}")
    logging.info(f"[Project Root] ROOT_DIR = {ROOT_DIR}")
    # logging.info(f"[Data Directory] TRAIN_ROOT_DIR = {TRAIN_ROOT_DIR}")
    # logging.info(f"Best Code Directory = {best_code_dir}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using model: {cfg.model}")
    logging.info(f"Evolving Module Type:{cfg.module_to_evo}")
    # Initialize LLM Client 
    client = init_client(cfg)
    
    torch.cuda.empty_cache()
    # Initialize Evolutor
    if cfg.model == 'pomo':
        tsp_evolver = POMO_EVO(cfg, root_dir = ROOT_DIR, generator_llm=client, debug_mode=DEBUG_MODE)
    elif cfg.model == "lehd":
        tsp_evolver = LEHD_EVO(cfg, root_dir = ROOT_DIR, generator_llm=client, debug_mode=DEBUG_MODE)
        
    else:
        raise NotImplementedError("parameter 'model' is missing in cofig file!")
        

    # Run Evolution Process
    elitist_module = tsp_evolver.evolve()

    json_file_name = f"elitist{cfg.module_to_evo}.json"
    
    logging.info(f"Elitist Modules:{elitist_module}")
    logging.info(f"[✓] Evolution completed.")
    # re-Save Final Best Module and Scores
    best_code_path = Path(ROOT_DIR, "best_code/"+json_file_name)
    with open(best_code_path, "w") as f:
      json.dump(elitist_module,f,indent=2)

    
    logging.info(f"[✓] Evolution completed. Best code saved to: {best_code_path}")


    
def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info("Cuda is Available:{}".format(torch.cuda.is_available()))
    
if __name__ == "__main__":
    _print_config()
    main()



