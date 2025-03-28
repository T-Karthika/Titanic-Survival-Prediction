# 🚢 Titanic Survival Prediction  

## 📌 Project Overview  
This project builds a Machine Learning model to predict whether a passenger survived the **Titanic disaster** based on various features like age, gender, ticket class, and fare.  

We use **data visualization, preprocessing, and classification models** to analyze the dataset and improve prediction accuracy.  

## 📂 Project Structure  
```
Titanic-Survival-Prediction/
├── data/                  
├── dataset/              
├── venv/             
├── README.md              
├── requirements.txt      
├── .gitignore           
```

## 🛠️ Installation & Setup  
1. **Clone the repository**  
   ```sh
   git clone https://github.com/T-Karthika/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```
2. Create a virtual environment (optional but recommended)
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. Install dependencies 
   ```sh
   pip install -r requirements.txt
   ```

## 🗃️ Dataset  
The dataset is sourced from the [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data).  

### **Features Used:**  
- *Pclass*: Ticket class (1st, 2nd, 3rd)  
- *Sex*: Gender (Male/Female)  
- *Age*: Passenger's age  
- *Fare*: Ticket fare  
- *Embarked*: Port of embarkation  
- *SibSp/Parch*: Number of siblings/spouses/parents/children aboard  
- *Survived (Target Variable)*: 1 = Survived, 0 = Did not survive  

## 📊 Data Visualization  
We analyzed survival rates based on **gender, class, and fare distribution** using `seaborn` and `matplotlib`.  

![Survival Rate by Gender](data/survival_by_gender.png)  

## 🤖 Model Training  
- Used **Logistic Regression** for classification  
- Handled missing values and performed **feature encoding**  
- Evaluated model using **accuracy, precision, recall, and F1-score**  


## 🚀 Future Improvements  
- Test with different ML models (Random Forest, SVM, etc.)  
- Implement **Hyperparameter tuning** for better accuracy  
- Deploy as a **Flask/Django web app**  

## 📞 Contact  
📧 Email: [2200030109cseh@gmail.com]
📌 GitHub: [T-Karthika](https://github.com/T-Karthika)

