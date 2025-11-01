echo "Script is start..."
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_PATH="$BASE_DIR/yolo-deneme1"
source "$VENV_PATH/bin/activate"
echo "Venv is active..."

python3 "$BASE_DIR/program.py"
echo "program is finished..."

deactivate
echo "VENV is deactivated."
