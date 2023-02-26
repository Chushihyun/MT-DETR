export CUDA_VISIBLE_DEVICES=0
config_path=mt_detr_c+l+r

python tools/train.py \
    configs/mt_detr/${config_path}.py  \
    --work-dir work_dir/${config_path} \
    

python tools/test.py \
    configs/mt_detr/${config_path}.py \
    checkpoint/model/${config_path}.pth \
    --eval bbox \

weather_list=(test_clear_day test_clear_night light_fog_day light_fog_night dense_fog_day dense_fog_night snow_day snow_night)

for w in "${weather_list[@]}"
do
    python tools/test.py \
        configs/mt_detr/${config_path}.py \
        checkpoint/model/${config_path}.pth \
        --eval bbox \
        --weather ${w}
done
