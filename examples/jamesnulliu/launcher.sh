set -e

unset ROCR_VISIBLE_DEVICES
export HF_HOME="/home/shared"

FILE_PATH=$(readlink -f "$0")
DIR_PATH=$(dirname "$FILE_PATH")

bash "$DIR_PATH/entropy2token.py"