 befor run training  should in terminal 
($env:TRANSFORMERS_NO_TF="1"
$env:USE_TF="0"
uv run python src/models/train_model.py
)


to run traun_model.py 
(uv run python src/models/train_model.py) in terminal

upload file \src\models\micropython_foresp32.py to your esp32
and check the port COM 
we used IOT mqtt panel  in phone ande used Narada to creat local proker
and chack the IP of proker