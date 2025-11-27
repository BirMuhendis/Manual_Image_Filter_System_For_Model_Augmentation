echo "Setup is start..."
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_PATH="$BASE_DIR/yolo-deneme1"
source "$VENV_PATH/bin/activate"
echo "VENV is activate..."

echo "Depenceties are installing..."
pip install --upgrade pip
pip install opencv-python numpy
echo "installing is finished."

deactivate
echo "please start the run.sh"
