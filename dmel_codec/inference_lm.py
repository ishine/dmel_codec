import hydra
import dmel_codec
from omegaconf import DictConfig
import torchaudio
from dmel_codec.utils.logger import RankedLogger
from dmel_codec.utils.print_config import print_config_tree
from dmel_codec.models.lm_lit_modules import MusicLLM
root_path = dmel_codec.__path__[0]
logger = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path=f"{root_path}/config/lm", config_name="lm_inference.yaml")
def main(config: DictConfig):
    print_config_tree(config)
    device = config.device
    logger.info(f"Using device: {device}")
    
    # 初始化模型
    model: MusicLLM = hydra.utils.instantiate(config.model)
    model = model.to(device)
    model.eval()

    logger.info("Model set to evaluation mode, ready to inference")
    predict_audio = model.inference_by_text_prompt(config)
    torchaudio.save("/home/wuzhiyue/dmel_codec-wzy_code/predict_audio_0_5B.wav", predict_audio, 24000)
    logger.info(f"Inference done, predict_audio shape: {predict_audio.shape}")

if __name__ == "__main__":
    main()