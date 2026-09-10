#!/bin/bash
# OpenCSLR 快速验证脚本
# 用于验证安装是否正确，并运行一个最小化的训练测试

set -e  # 遇到错误立即退出

echo "=================================="
echo "OpenCSLR Installation Verification"
echo "=================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查 Python 版本
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" == 3.7* ]] || [[ "$PYTHON_VERSION" == 3.8* ]]; then
    echo -e "  ${GREEN}✓ Python version OK${NC}"
else
    echo -e "  ${YELLOW}⚠ Warning: Python 3.7 recommended, you have $PYTHON_VERSION${NC}"
fi
echo ""

# 2. 检查关键依赖
echo "Step 2: Checking dependencies..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" || { echo -e "${RED}✗ PyTorch not found${NC}"; exit 1; }
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')" || { echo -e "${RED}✗ NumPy not found${NC}"; exit 1; }
python -c "import yaml; print(f'  PyYAML: installed')" || { echo -e "${RED}✗ PyYAML not found${NC}"; exit 1; }
echo -e "  ${GREEN}✓ Core dependencies OK${NC}"
echo ""

# 3. 检查 CUDA
echo "Step 3: Checking CUDA availability..."
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA devices: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"
echo ""

# 4. 检查目录结构
echo "Step 4: Checking directory structure..."
if [ -d "core" ] && [ -f "core/main.py" ]; then
    echo -e "  ${GREEN}✓ Directory structure OK${NC}"
else
    echo -e "  ${RED}✗ Please run this script from OpenCSLR root directory${NC}"
    exit 1
fi
echo ""

# 5. 检查配置文件
echo "Step 5: Checking configuration files..."
if [ -f "core/configs/unified_phoenix2014.yaml" ]; then
    echo -e "  ${GREEN}✓ Config files found${NC}"
else
    echo -e "  ${YELLOW}⚠ Warning: Unified config templates not found${NC}"
fi
echo ""

# 6. 测试导入
echo "Step 6: Testing module imports..."
cd core
python -c "
import sys
try:
    from manager.argument_manager import ArgumentManager
    from manager.config_manager import ConfigManager
    from manager.experiment_manager import ExperimentManager
    print('  ${GREEN}✓ Core managers import OK${NC}')
except Exception as e:
    print('  ${RED}✗ Import failed: ' + str(e) + '${NC}')
    sys.exit(1)

try:
    from utils.seed_utils import set_seed
    print('  ${GREEN}✓ Utility modules import OK${NC}')
except ImportError:
    print('  ${YELLOW}⚠ Utils modules not found (optional)${NC}')
" || { echo -e "${RED}✗ Import test failed${NC}"; exit 1; }
cd ..
echo ""

# 7. 验证协议合规性（如果脚本存在）
echo "Step 7: Validating protocol compliance..."
if [ -f "scripts/check_protocol_compliance.py" ]; then
    if [ -f "core/configs/unified_phoenix2014.yaml" ]; then
        python scripts/check_protocol_compliance.py core/configs/unified_phoenix2014.yaml --no-verbose
    else
        echo -e "  ${YELLOW}⚠ Config file not found, skipping${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠ Compliance checker not found, skipping${NC}"
fi
echo ""

# 8. 样本统计 / 结果汇总集成测试（纯 CPU，无 GPU 与数据集）
echo "Step 8: Running stats integration test..."
if [ -f "core/tests/test_stats_integration.py" ]; then
    (cd core && python tests/test_stats_integration.py) \
        || { echo -e "${RED}✗ Stats integration test failed${NC}"; exit 1; }
else
    echo -e "  ${YELLOW}⚠ Test file not found, skipping${NC}"
fi
echo ""

# 9. 总结
echo "=================================="
echo "Verification Summary"
echo "=================================="
echo -e "${GREEN}✓ Installation verification completed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset (see docs/dataset_preparation.md)"
echo "  2. Run a smoke test:"
echo "     cd core"
echo "     python main.py --config configs/exp.yaml --exp baseline --num_epoch 1 --work-dir /tmp/smoke_test"
echo "  3. Check the documentation:"
echo "     - README.md for quick start"
echo "     - INSTALL.md for detailed installation"
echo "     - docs/PROTOCOLS.md for experiment protocols"
echo ""
echo "For support, open an issue at:"
echo "  https://github.com/immc-lab/OpenCSLR/issues"
echo ""
