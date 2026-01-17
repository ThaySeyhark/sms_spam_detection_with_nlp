# SMS Spam Detection with NLP

* [Royal University of Phnom Penh](https://www.rupp.edu.kh/graduate/mite/index.php?page=Curriculum)
* Code Reference: https://www.kaggle.com/code/dktalaicha/sms-spam-detection-with-nlp
* Dataset: https://www.kaggle.com/code/dktalaicha/sms-spam-detection-with-nlp/?select=spam.csv
* Reference Paper: https://tijer.org/tijer/papers/TIJER2506009.pdf

### **Project Setup and Workflow**

1. **Use Jupyter Notebook**
    The project requires [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) for running and training models.

2. **Create a Virtual Environment**
    In the project directory, create a Python virtual environment (**Note**: This project uses `v3.13.7`):

   ```sh
   python3.13 -m venv env
   ```

3. **Train the SMS Spam Detection Model**
    Open and run the training notebook:

```sh
sms_spam_detection_with_nlp.ipynb
```

4. **Model Artifacts After Training**
    Once training is complete, a `models` folder will be generated containing:
   - `count_vectorizer.pkl` (Bag of Words)
   - `tfidf_transformer.pkl` (TF-IDF)
   - `ig_selector.pkl` (Information Gain)
   - `spam_detect_model.pkl` (Naive Bayes Model)

    ***Note**: All the models listed above are required because a new test data must follow the same workflow as the training process.*
   
5. **Predict Using Saved Models (Deployment)**

For prediction on a new data, use:

- `app_terminal.ipynb` (terminal-based interface)

- or `app.py` (web UI — requires installing Streamlit):

  ```
  ! pip install streamlit
  ```

6. **Run the Application**
    After launching the application, you can input any new SMS message to detect whether it is spam or not.

Thank you!

#### URL for testing:

https://sms-spam-detection-with-nlp.onrender.com/
