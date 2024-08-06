# llm20

Some work to solve the Kaggle [LLM 20 Questions](https://www.kaggle.com/competitions/llm-20-questions)

## Submission

0. put model weights in `submission/input` folder
1. compress work
    ```bash
    tar -cf - -C submission . | pigz --fast | pv > submission.tar.gz
    ```

2. submit
    ```bash title=""
    kaggle competitions submit -c llm-20-questions -f submission.tar.gz -m "<submission_message>"
    ```
