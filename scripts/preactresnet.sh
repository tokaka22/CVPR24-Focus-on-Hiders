# cd /code/HFAT

name=Focus_preact
model_name=hfat
savedir=./res/main_${model_name}/

mkdir -p $savedir

file_to_check=$savedir/run_$run_name.txt

if [ -f "$file_to_check" ]; then
    rm "$file_to_check"
    echo "deleted"
else
    echo "files donot exist"
fi

run_name=00
wd_name=${model_name}_${run_name}

echo $savedir
echo $wd_name

CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --model PreActResNet18 \
    --epsilon 8 \
    --pgd_alpha 2 \
    --attack_iters 10 \
    --eps_gamma 0.01 \
    --chkpt_iters 20 \
    --fname $savedir \
    --proj_name $name \
    --name $wd_name \
    --wd_offline 0 \
    --lr_proxy_max 0.05 \
    --beta_r 0.0 \
    --beta_u 1.0 \
    --mean 0.4 \
    --std_dev 0.001 \
    --lt 1.5 \
    2>&1 | tee $savedir/$run_name.txt

