#! /bin/bash
gpu=$1
device=$2
outdir=$3 
ckpt=$4
for ((j=0;j<100;j+=9))
do
if [ $j -eq 99 ]; then
        echo $j
        mkdir -p "$outdir"/test_"$j"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $j --outdir "$outdir"/test_"$j"_pdb --ckpt $ckpt > "$outdir"/test_"$j"_pdb/sample_molecule.log 2>&1 &
        pid1=$!
        wait $pid1
else
        mkdir -p "$outdir"/test_"$j"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $j --outdir "$outdir"/test_"$j"_pdb --ckpt $ckpt > "$outdir"/test_"$j"_pdb/sample_molecule.log 2>&1 &
        pid1=$!
        mkdir -p "$outdir"/test_"$[$j+1]"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $[$j+1] --outdir "$outdir"/test_"$[$j+1]"_pdb --ckpt $ckpt > "$outdir"/test_"$[$j+1]"_pdb/sample_molecule.log 2>&1 &
        pid2=$!
        mkdir -p "$outdir"/test_"$[$j+2]"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $[$j+2] --outdir "$outdir"/test_"$[$j+2]"_pdb --ckpt $ckpt > "$outdir"/test_"$[$j+2]"_pdb/sample_molecule.log 2>&1 &
        pid3=$!
        mkdir -p "$outdir"/test_"$[$j+3]"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $[$j+3] --outdir "$outdir"/test_"$[$j+3]"_pdb --ckpt $ckpt > "$outdir"/test_"$[$j+3]"_pdb/sample_molecule.log 2>&1 &
        pid4=$!
        mkdir -p "$outdir"/test_"$[$j+4]"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $[$j+4] --outdir "$outdir"/test_"$[$j+4]"_pdb --ckpt $ckpt > "$outdir"/test_"$[$j+4]"_pdb/sample_molecule.log 2>&1 &
        pid5=$!
        mkdir -p "$outdir"/test_"$[$j+5]"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $[$j+5] --outdir "$outdir"/test_"$[$j+5]"_pdb --ckpt $ckpt > "$outdir"/test_"$[$j+5]"_pdb/sample_molecule.log 2>&1 &
        pid6=$!
        mkdir -p "$outdir"/test_"$[$j+6]"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $[$j+6] --outdir "$outdir"/test_"$[$j+6]"_pdb --ckpt $ckpt > "$outdir"/test_"$[$j+6]"_pdb/sample_molecule.log 2>&1 &
        pid7=$!
        mkdir -p "$outdir"/test_"$[$j+7]"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $[$j+7] --outdir "$outdir"/test_"$[$j+7]"_pdb --ckpt $ckpt > "$outdir"/test_"$[$j+7]"_pdb/sample_molecule.log 2>&1 &
        pid8=$!
        mkdir -p "$outdir"/test_"$[$j+8]"_pdb
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u sample_molecule.py --device $device --data_id $[$j+8] --outdir "$outdir"/test_"$[$j+8]"_pdb --ckpt $ckpt > "$outdir"/test_"$[$j+8]"_pdb/sample_molecule.log 2>&1 &
        pid9=$!
        wait $pid1
        wait $pid2
        wait $pid3
        wait $pid4
        wait $pid5
        wait $pid6
        wait $pid7
        wait $pid8
        wait $pid9
fi
done


