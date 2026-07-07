qwen_image_series = [
    {
        "model_hash": "0319a1cb19835fb510907dd3367c95ff",
        "model_name": "qwen_image_dit",
        "model_class": "diffsynth.models.qwen_image_dit.QwenImageDiT",
    },
    {
        "model_hash": "8004730443f55db63092006dd9f7110e",
        "model_name": "qwen_image_text_encoder",
        "model_class": "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.qwen_image_text_encoder.QwenImageTextEncoderStateDictConverter",
    },
    {
        "model_hash": "ed4ea5824d55ec3107b09815e318123a",
        "model_name": "qwen_image_vae",
        "model_class": "diffsynth.models.qwen_image_vae.QwenImageVAE",
    },
    {
        "model_hash": "073bce9cf969e317e5662cd570c3e79c",
        "model_name": "qwen_image_blockwise_controlnet",
        "model_class": "diffsynth.models.qwen_image_controlnet.QwenImageBlockWiseControlNet",
    },
    {
        "model_hash": "a9e54e480a628f0b956a688a81c33bab",
        "model_name": "qwen_image_blockwise_controlnet",
        "model_class": "diffsynth.models.qwen_image_controlnet.QwenImageBlockWiseControlNet",
        "extra_kwargs": {"additional_in_dim": 4},
    },
    {
        "model_hash": "49ae821b996dc7e19a5e8d9d4f3c5315",
        "model_name": "qwen_image_connector",
        "model_class": "diffsynth.models.qwen_image_connector.QwenImageConnector",
        "extra_kwargs": {"input_dim": 768, "num_layers": 6},
    },
    {
        "model_hash": "7dd3640ddce93380be19b6c86f9e7e64",
        "model_name": "qwen_image_connector",
        "model_class": "diffsynth.models.qwen_image_connector.QwenImageConnector",
        "extra_kwargs": {"num_layers": 6},
    },
    {
        "model_hash": "7b45ff1c1703e389ef61c8ab12babc4d",
        "model_name": "qwen_image_learnable_query",
        "model_class": "diffsynth.models.qwen_image_learnable_query.QwenImageLearnableQuery",
        "extra_kwargs": {"query_length": 256},
    },
    {
        "model_hash": "5dbd9a25c517478a8d2f35b01da53e9a",
        "model_name": "qwen_image_extractor",
        "model_class": "diffsynth.models.qwen_image_extractor.QwenImageExtractor",
    },
    {
        "model_hash": "bc4b4d58554d31694168550738ae8286",
        "model_name": "qwen_image_rformer",
        "model_class": "diffsynth.models.qwen_image_rformer.RFormer",
    },
    {
        "model_hash": "aea189b86b5df7cffc8ec81d76fde9a5",
        "model_name": "qwen_image_rformer2D",
        "model_class": "diffsynth.models.qwen_image_rformer.RFormer2D",
    },
    {
        "model_hash": "bc4aeac1e11041ca661685e9ed1e3ead",
        "model_name": "qwen_image_mae",
        "model_class": "diffsynth.models.qwen_image_mae.QwenImageMAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.qwen_image_mae.QwenImageMAEStateDictConverter",
    },
]

MODEL_CONFIGS = qwen_image_series
