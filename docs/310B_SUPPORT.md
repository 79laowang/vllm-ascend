# Ascend 310B Series NPU Support for vLLM-Ascend

## Overview

This document describes support for the Huawei Ascend 310B series NPUs (310B1, 310B2, 310B3, 310B4) in vLLM-Ascend.

## Supported Devices

| Device | soc_version | Status |
|--------|-------------|--------|
| Ascend310B1 | 15 | ✅ Supported |
| Ascend310B2 | 33 | ✅ Supported |
| Ascend310B3 | 34 | ✅ Supported |
| Ascend310B4 | 35 | ✅ Supported |

## Hardware Specifications

- **AI Core:** 1
- **Vector Core:** 1
- **AI CPU:** 4
- **Memory:** ~15GB (varies by model)
- **Architecture:** Entry-level inference chip

## Known Limitations

### 1. Flash Attention Not Supported
The 310B series does not support Flash Attention operations. You must use eager attention:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation='eager'  # Required for 310B
)
```

Or set the environment variable:
```bash
export INF_NPU_DISABLE_FLASH_ATTENTION=True
```

### 2. Operator Compilation
On first use, operators need to be JIT compiled by the CANN toolkit. This can take several minutes. Subsequent runs will use the compiled operators.

### 3. CANN 8.1.RC1 Compatibility
Some operators may have compatibility issues with CANN 8.1.RC1. The following operators are known to have issues:
- `ConcatD` - May fail with certain data types
- `FlashAttentionScore` - Not supported (error code 561000)

## Setup

### Quick Setup

Use the provided setup script to configure your environment:

```bash
source ~/bin/vllm-ascend-310b-setup
```

### Manual Setup

Set these environment variables:

```bash
# Fix for protobuf ABI mismatch
export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/lib64/libascend_protobuf.so

# Ascend toolkit paths
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp

# Disable Flash Attention
export INF_NPU_DISABLE_FLASH_ATTENTION=True
```

### Persistent Setup

Add to your `~/.bashrc`:

```bash
# vLLM-Ascend 310B Series NPU Support
if [ -f "$HOME/bin/vllm-ascend-310b-setup" ]; then
    source "$HOME/bin/vllm-ascend-310b-setup" 2>/dev/null || true
fi
```

## Usage

### Using vLLM-Ascend

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model='Qwen/Qwen2-1.5B-Instruct',
    tensor_parallel_size=1,
    max_model_len=2048,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    max_tokens=256,
)

outputs = llm.generate(["Hello, who are you?"], sampling_params)
```

### Using Transformers Directly

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2-1.5B-Instruct',
    torch_dtype=torch.float16,
    device_map='npu:0',
    trust_remote_code=True,
    attn_implementation='eager'  # Required for 310B
)

inputs = tokenizer("Hello, who are you?", return_tensors='pt').to('npu:0')
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Device Detection

To verify 310B support:

```python
from vllm_ascend.utils import get_ascend_device_type, is_310b

print(f"Device type: {get_ascend_device_type()}")  # AscendDeviceType._310B
print(f"is_310b: {is_310b()}")  # True
```

## Performance Considerations

### Memory Usage
- Model memory usage: ~2-3GB for 1.5B models (fp16)
- Peak memory during inference: May reach 12-15GB
- Recommended max sequence length: 2048 tokens

### Performance Tips

1. **Use smaller models**: 310B is optimized for smaller models (1.5B-7B)
2. **Limit sequence length**: Keep max_model_len <= 2048 for best performance
3. **Use greedy decoding**: Set `do_sample=False` for faster inference
4. **Batch size**: Use smaller batch sizes (1-4) to avoid OOM

### Expected Performance

Approximate tokens/sec for Qwen2-1.5B:
- First run (with compilation): ~1-2 tokens/sec (compilation overhead)
- Subsequent runs: ~5-10 tokens/sec

*Note: Performance varies based on model size, sequence length, and batch size*

## Troubleshooting

### Error: "Op FlashAttentionScore does not has any binary"

**Solution:** Flash Attention is not supported on 310B. Use eager attention:
```bash
export INF_NPU_DISABLE_FLASH_ATTENTION=True
```

### Error: "undefined symbol: _ZNK14ascend_private8protobuf7Message11GetTypeNameEv"

**Solution:** This is a protobuf ABI mismatch. Set LD_PRELOAD:
```bash
export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/lib64/libascend_protobuf.so
```

### Error: "unsupported data type for this op"

**Solution:** Try using float16 instead of bfloat16:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16
    ...
)
```

### Error: "RuntimeError: Can not support soc_version"

**Solution:** Make sure you're using the latest vLLM-Ascend with 310B support. Check your device:
```bash
npu-smi info
```

## Contributing

To improve 310B support, please consider contributing:

1. **Operator optimization**: Add optimized operators for 310B
2. **Testing**: Add more test cases for 310B
3. **Documentation**: Improve this guide with your findings
4. **Bug reports**: Report issues with 310B support

## References

- [vLLM-Ascend GitHub](https://github.com/HuaweiAscend/vllm-ascend)
- [Ascend 310B4 Datasheet](https://www.hiascend.com/hardware/fixed-tensor-processor)
- [CANN Toolkit Documentation](https://www.hiascend.com/document)

## Changelog

### 2026-03-02
- Initial 310B support added
- Device detection for 310B1/310B2/310B3/310B4
- Configuration helper script
- Documentation created
