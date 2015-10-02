#!/usr/bin/env sh
# Create the classification $DB_TYPE inputs
# N.B. set the path to the dataset train + val data dirs

if [ $# -gt 8 ] || [ $# -lt 3 ]
then
  echo "Usage: $0 ROOTDIR name dataset [DB=lmdb] [shuffle=0] [resize=0] [check_size=1] [gray=0]"
  exit
fi

ROOTDIR=$1
name=$2
dataset=$3
DB_TYPE=lmdb
if [ $# -ge 4 ]
then
  DB_TYPE=$4
fi
shuffle=0
if [ $# -ge 5 ]
then
  shuffle=$5
fi
resize=0
if [ $# -ge 6 ]
then
  resize=$6
fi
check_size=1
if [ $# -ge 7 ]
then
  check_size=$7
fi
gray=0
if [ $# -ge 8 ]
then
  gray=$8
fi

EXAMPLE=$ROOTDIR/$name/$DB_TYPE
DATA=data/$name
TOOLS=build/tools

DATA_ROOT=$ROOTDIR/$name/

if [ ! -d $EXAMPLE ]
then
  mkdir -p $EXAMPLE
fi

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=$resize
if [ $RESIZE -ne 0 ]
then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
  echo "Set the DATA_ROOT variable in create_segment.sh to the path" \
       "where the classification data is stored."
  exit 1
fi

echo "Creating $name $dataset $DB_TYPE..."

if [ -d $EXAMPLE/"$name"_"$dataset"_$DB_TYPE ]
then
  rm -r $EXAMPLE/"$name"_"$dataset"_$DB_TYPE
fi

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=$shuffle \
    --check_size=$check_size \
    --gray=$gray \
    --backend=$DB_TYPE \
    $DATA_ROOT \
    $DATA/$dataset.txt \
    $EXAMPLE/"$name"_"$dataset"_$DB_TYPE

if [ ! -d examples/$name ]
then
  mkdir examples/$name
fi
rm -f examples/$name/"$name"_"$dataset"_$DB_TYPE
ln -s $EXAMPLE/"$name"_"$dataset"_$DB_TYPE examples/$name/

echo "Done."
