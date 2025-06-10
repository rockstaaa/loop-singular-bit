# 🚀 Loop Singular Bit - Quick Start Guide

**Get started with Loop Singular Bit in 5 minutes!**

## ⚡ One-Command Installation

```bash
# Clone and setup
git clone https://github.com/rockstaaa/loop-singular-bit.git
cd loop-singular-bit
pip install -r requirements.txt

# Test the system (30 seconds)
python simple_test.py
```

## 🎯 Instant Demo

```python
# Run this to see compression working immediately
python -c "
import sys, os
sys.path.insert(0, 'src/loop_singular_bit')
from quantization import OutlierPreservingQuantizer
import torch

print('🔥 Loop Singular Bit - Live Demo')
quantizer = OutlierPreservingQuantizer(outlier_ratio=0.02)
tensor = torch.randn(1000, 1000) * 0.02
result = quantizer.quantize(tensor, 'demo')

print(f'✅ Compressed: {result[\"compression_ratio\"]:.2f}× ratio')
print(f'✅ Quality: {result[\"quality_error_percent\"]:.2f}% error')
print('🎉 System working!')
"
```

## 🎮 Interactive Examples

### 1. **Basic Compression**
```bash
python examples/basic_compression.py
```

### 2. **Live System Demo**
```bash
python working_demo.py
```

### 3. **Complete System Test**
```bash
python final_system_test.py
```

## 🌐 Web Interface (LIVE NOW!)

**🎮 HuggingFace Spaces:** [loop-singular-bit](https://huggingface.co/spaces/rockstaaa/loop-singular-bit)

Available features:
- ✅ **Compression Demo** → Test real tensor compression
- ✅ **Text Generation** → Generate text with compressed models
- ✅ **System Information** → View capabilities and status
- ✅ **Interactive Documentation** → Learn while you explore

## 📱 Usage Options

### For Developers
- **GitHub Repository**: Full source code access
- **Local Installation**: Complete control
- **API Integration**: Embed in your projects

### For Researchers  
- **Research Papers**: Complete technical documentation
- **Experimental Data**: Real measurement results
- **Proof Files**: Verification evidence

### For General Users
- **Web Interface**: Simple drag-and-drop (planned)
- **Pre-compressed Models**: Ready-to-use models
- **Online Playground**: No installation needed (planned)

## 🎯 Next Steps

1. **Try the Quick Demo** (above) - 30 seconds
2. **Run Full Tests** - See all capabilities
3. **Read Documentation** - Understand the technology
4. **Join Development** - Contribute improvements

## 🔗 Links

- **Repository**: https://github.com/rockstaaa/loop-singular-bit
- **Documentation**: `/docs/technical_guide.md`
- **Research**: `/research/` directory
- **Examples**: `/examples/` directory

**🚀 Ready to compress 675B models on 8GB RAM? Start now!**
