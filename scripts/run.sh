#!/bin/bash
set -e
for ARGUMENT in "${@:1}"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    if [[ $KEY == --* ]]
    then
      KEY=${KEY:2}
    else
      KEY=${KEY}
    fi
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    declare $KEY="$VALUE"
    export $KEY="$VALUE"
    if [[ $KEY == name ]]
    then
      name=$VALUE
    elif [[ $ARGUMENT == --* ]] # * is used for pattern matching
    then
      python_arg="$python_arg $ARGUMENT"
    fi
done
#${prepropath} next-models/actev_single_model model
prepropath=${prepropath:-actev_preprocess_${obs_len:-8}_${pred_len:-12}}
save_folder=${save_folder:-"next-models/actev_single_model"}
modelname=${modelname:-"model"}

add_kp=${add_kp:-1}
add_activity=${add_activity:-1}
multi_decoder=${multi_decoder:-1}
preload_features=${preload_features:-1}
embed_traj_label=${embed_traj_label:-0}
runId=${runId:-0}

base_args="${prepropath} ${save_folder} ${modelname} \
  --runId ${runId} \
  --is_actev \
  $([ $add_kp == 1 ] && echo "--add_kp" || echo "" ) \
  $([ $add_activity == 1 ] && echo "--add_activity" || echo "" ) \
  --person_feat_path ${person_feat_path:-"next-data/actev_personboxfeat"} \
  $([ $multi_decoder == 1 ] && echo "--multi_decoder" || echo "" ) \
  $([ $preload_features == 1 ] && echo "--preload_features" || echo "" ) \
  $([ $embed_traj_label == 1 ] && echo "--embed_traj_label" || echo "" ) \
  --obs_len ${obs_len:-8} \
  --pred_len ${pred_len:-12}"

echo $base_args
#exit 0

if [ ${run_mode:-all} = "all" ] || [ ${run_mode:-all} = "train" ]; then
  echo -------------------------train ----------------------
python3 code/train.py ${base_args} --message "${message:-no_message}" --group=${group:-default}
fi


if [ ${run_mode:-all} = "all" ] || [ ${run_mode:-all} = "test_single" ]; then
  echo -------------------------Test Single ----------------------
python3 code/test.py ${base_args} --save_output=single_${modelname}.traj.p  --load_best
fi