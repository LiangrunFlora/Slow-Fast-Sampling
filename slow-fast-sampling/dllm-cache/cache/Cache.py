import torch
from collections import defaultdict
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FeatureCache(metaclass=Singleton):
    gen_interval_steps: int
    prompt_interval_steps: int
    cfg_interval_steps: int 
    prompt_length: int
    transfer_ratio:float
    __cache: defaultdict  # 统一缓存，区分 prompt 和 gen
    __step_counter: defaultdict

    @classmethod
    def new_instance(cls, prompt_interval_steps: int = 1, gen_interval_steps: int = 1,cfg_interval_steps: int = 1,transfer_ratio:float = 0.0) -> "FeatureCache":
        ins = cls()
        setattr(ins, "prompt_interval_steps", prompt_interval_steps)
        setattr(ins, "gen_interval_steps", gen_interval_steps)
        setattr(ins, "cfg_interval_steps", cfg_interval_steps)
        setattr(ins, "transfer_ratio", transfer_ratio)
        ins.init()
        return ins

    def init(self) -> None:
        # 使用单一缓存结构，通过 cache_type 区分 prompt 和 gen
        # self.__cache = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.__cache = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        self.__step_counter = defaultdict(lambda: defaultdict(lambda: 0))

    def reset_cache(self, prompt_length: int = 0) -> None:
        self.init()
        torch.cuda.empty_cache()
        self.prompt_length = prompt_length
        self.cache_type = "no_cfg"

    def set_cache(self, layer_id: int, feature_name: str, features: torch.Tensor, cache_type: str,save_flag:bool = False) -> None:
        self.__cache[self.cache_type][cache_type][layer_id][feature_name] =  {0: features}
        if save_flag:
            if feature_name=="kv_cache":
                torch.save(features["k"], f"./cache_save/step{self.current_step}_layer{layer_id}_k_{cache_type}.pt")
                torch.save(features["v"], f"./cache_save/step{self.current_step}_layer{layer_id}_v_{cache_type}.pt")
                print(f"Saved cache to ./cache_save/step{self.current_step}_layer{layer_id}_k_{feature_name}.pt")
                print(f"Saved cache to ./cache_save/step{self.current_step}_layer{layer_id}_V_{feature_name}.pt")
            else:
                torch.save(features, f"./cache_save/step{self.current_step}_layer{layer_id}_{feature_name}_{cache_type}.pt")
                print(f"Saved cache to ./cache_save/step{self.current_step}_layer{layer_id}_token_{cache_type}.pt")

    def get_cache(self, layer_id: int, feature_name: str, cache_type: str) -> torch.Tensor:
        output = self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]
        return output

    def update_step(self, layer_id: int) -> None:
        self.__step_counter[self.cache_type][layer_id] += 1

    def refresh_gen(self, layer_id: int = 0) -> bool:
        return (self.current_step-1) % self.gen_interval_steps == 0 
    
    def refresh_prompt(self, layer_id: int = 0) -> bool:
        return (self.current_step-1) % self.prompt_interval_steps == 0 
        
    def refresh_cfg(self, layer_id: int = 0) -> bool:
        return (self.current_step-1) % self.cfg_interval_steps == 0 or self.current_step <=5
    @property
    def current_step(self) -> int:
        return max(list(self.__step_counter[self.cache_type].values()), default=1)
    def __repr__(self):
        return f"USE FeatureCache"


