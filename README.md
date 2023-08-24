# End-to-end LLM Deployments in Databricks 😎

Just messing around with different ways of quickly deploying and interacting with LLMs served on Databricks...

A couple things:
* This POC uses Langchain, so you can quickly swap out the LLM you wish to use by simply redefining `LLM` in `app.py`
* To run the Gradio chat UI:
  
  ```bash
  python app.py
  ```

![screenshot](./docs/screenshot.png)