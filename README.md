# function-calling

The data we need to prepare in advance is the **function_list.jsonl** data, which contains function descriptions, parameters, etc as follows:

:bulb: **function_list.jsonl**
```
[{
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": "특정 도시의 지정된 기간 동안의 날씨 예보를 가져올 수 있다. 온도, 강수량, 바람 등의 상세 정보를 제공하며, 지정된 기간 내 최고/최저 기온을 확인할 수 있다.\n리턴객체: WeatherForecast",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "날씨 정보를 확인할 도시 이름"
                },
                "start_date": {
                    "type": "string",
                    "description": "yyyy-mm-dd 형식으로 시작 날짜"
                },
                "end_date": {
                    "type": "string",
                    "description": "yyyy-mm-dd 형식으로 종료 날짜"
                }
            },
            "required": ["city", "start_date", "end_date"]
        }
    }
} ... 
]
```

With this prepared data, we create **function_call_data.jsonl**, which is the training data. **function_call_data.jsonl** is created using function_list and openai api. It consists of user query (input) and function with parameters (output), as in the following example.

:bulb: **function_call_data.jsonl**

```
[
    {
        "input": "서울의 올해 1월 날씨를 알려줄래? 시작일은 2023년 1월 1일이고, 종료일은 1월 31일이야.",
        "output": [
            {
                "tool": "get_weather_forecast",
                "tool_input": "{\"city\": \"서울\", \"start_date\": \"2023-01-01\", \"end_date\": \"2023-01-31\"}"
            }
        ]
    },
    {
        "input": "2023년 2월 동안 뉴욕의 날씨가 어땠는지 알 수 있을까?",
        "output": [
            {
                "tool": "get_weather_forecast",
                "tool_input": "{\"city\": \"뉴욕\", \"start_date\": \"2023-02-01\", \"end_date\": \"2023-02-28\"}"
            }
        ]
    }, ...
]
```

### prepare train data
```
python data/genererate_data.py
python data/preprocess_data.py
```

### lora finetuning
```
python train/main.py
```

### create an Excel file for qualitative evaluation
```
python test/lora_merge.py
python test/test.py
```
