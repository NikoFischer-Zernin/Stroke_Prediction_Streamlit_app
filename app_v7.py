import streamlit as st
import pandas as pd
import pickle 
#from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder

###########Page Configuration################
st.set_page_config(
    page_title= "Stroke Risk Web App",
    page_icon= "🏥",
    layout= "wide"
    )

#Page Title
st.title("Stroke Risk Assessment Web App")
st.markdown("Using Big Data to enhance the insurance industry!")

#load dataset
@st.cache()
def load_data():
    data = pd.read_csv("trainingsdaten.csv")
    return data.dropna()

@st.cache(allow_output_mutation=True)
def load_model():
    filename = "tuned_model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model


data = load_data()
model = load_model()


#Input Section

st.header("Data Input")
st.subheader("Please fill out the following questions in order to receive an assessment")



#columns for user input (three strongest variables)
row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

age = row1_col1.number_input('Age:', min_value = 0,
                             max_value = 82,
                             value = 40)
                            

gender = row1_col2.selectbox(
     'Gender:',
     ("Female", "Male", "Other"))


married = row1_col3.selectbox(
     'Has the customer ever been married?',
     ("Yes", "No"))


# second column for user input 
row2_col1, row2_col2, row2_col3 = st.columns([1,1,1])

smoker = row2_col1.selectbox(
     'Smoking Status:',
     ( "never smoked", "formerly smoked", "active smoker","unknown"))

hypertension = row2_col2.selectbox(
    "Does the customer have any history of Hypertension?",
    ("Yes", "No"))


heart = row2_col3.selectbox(
    "Does the customer have any history of heart disease?",
    ("Yes", "No"))

# third column for user input 
row3_col1, row3_col2, row3_col3 = st.columns([1,1,1])
glucose = row3_col1.slider("Select the average glucose level over past 2-3 months:",
                    min_value = 50.0,
                    max_value = 300.0,
                    value = 175.0)



work_type = row3_col2.selectbox(
    "Select the customers current type of work:",
    ("Children",
    "Never Worked",
    "Private Sector",
    "Self-employed",
    "Government Job"))

residence_type = row3_col3.selectbox(
    "Select the customers current type of residence:",
    ("rural",
    "urban"))
row4_col1, row4_col2 = st.columns([1,2])

if row4_col1.checkbox('Calculate BMI', False):
    row5_col1, row5_col2 = st.columns([1,1])
    
    height = row5_col1.slider('Height (in Centimeters)',
                              min_value = 100.0,
                              max_value = 210.0,
                              value = 180.0)
    height = height/100

    weight = row5_col2.slider('Weight (in Kilograms)',
                              min_value = 50.0,
                              max_value = 160.0,
                              value = 80.0)
    bmi = round(weight/(height**2),2)
    st.write(f"BMI: {bmi}") 
    
else:
    bmi = row4_col2.slider(
            'Select the customers BMI:',
            12.0,
            42.0,
            27.0)
    
#st.write(f"BMI: {bmi}") 
   

def predict_stroke_risk():
    cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Female', 'gender_Male', 'gender_Other', 'ever_married_No',
       'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
       'work_type_Private', 'work_type_Self-employed', 'work_type_children',
       'Residence_type_Rural', 'Residence_type_Urban',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes', 'smoking_status_unknown']
    df = pd.DataFrame(columns = cols)
    nulls = []
    for i in range (len(cols)):
        nulls.append(0)
    df.loc[0] = nulls
    df['age'] = age
    df['bmi'] = bmi
    df['avg_glucose_level'] = glucose
    if gender == 'Female':
        df['gender_Female'] = 1
    elif gender == 'Male':
        df['gender_Male'] = 1
    elif gender == 'Other':
        df['gender_Other'] = 1
    
    if married == 'Yes':
        df['ever_married_Yes'] = 1
    elif married == 'No':
        df['ever_married_No'] = 1
    
    if smoker == 'never smoked':
        df['smoking_status_never smoked'] = 1
    elif smoker == 'formerly smoked':
        df['smoking_status_formerly smoked'] = 1
    elif smoker == 'active smoker':
        df['smoking_status_smokes'] = 1
    elif smoker == 'unknown':
        df['smoking_status_unknown'] = 1
    
    if hypertension == 'Yes':
        df['hypertension'] = 1
    
    
    if heart == 'Yes':
        df['heart_disease'] = 1
        
    if work_type == 'Children':
        df['work_type_children'] = 1
    elif work_type == "Never Worked":
        df['work_type_Never_worked'] = 1
    elif work_type == "Private Sector":
        df['work_type_Private'] = 1
    elif work_type == 'Self-employed':
        df['work_type_Self-employed'] = 1
    elif work_type == 'Government Job':
        df['work_type_Govt_job'] = 1
    
    if residence_type == 'rural':
        df['Residence_type_Rural'] = 1
    elif residence_type == 'urban':
        df['Residence_type_Urban'] = 1
    
    
    return df



df = predict_stroke_risk()


preds = model.predict(df)
#row4_col2.markdown(f' The predicted risk is: {preds[0]}')
#################################################################

def show_result(preds):
    for value in preds:
        value = float(value)
        if value <= 0.05:
            st.markdown("According to the given input data, there is currently no increased stroke risk in the next 10 years.")
        
        elif value > 0.05 and value <= 0.1:
            st.markdown("According to the given input data, there is a slighlty increased stroke risk in the next 10 years.")
                     
        elif value > 0.1 and value <= 0.15:
            st.markdown("According to the given input data, there is a moderately increased stroke risk in the next 10 years.")
            
        elif value > 0.15 and value <= 0.2:
            st.markdown("According to the given input data, there is a significantly increased stroke risk in the next 10 years.")
            
        elif value > 0.2:
            st.markdown("According to the given input data, there is a dangerously increased stroke risk in the next 10 years.")
                     
st.subheader("Risk Assessment: ")
st.markdown("ATTENTION: This is prediction based upon limited data purely for estimative purposes.")                   
show_result(preds)

###################################################
#correlate the risk values to a premium class
def insurance_class(preds, fromBrowser):
    premium = []
    for value in preds:
        value = float(value)
        if value <= 0.05:
            premium.append(1)
               
        elif value > 0.05 and value <= 0.1:
            premium.append(1)
              
        elif value > 0.1 and value <= 0.15:
            premium.append(2)
            
        elif value > 0.15 and value <= 0.2:
            premium.append(2)
            
        elif value > 0.2:
            premium.append(3)
            
        if fromBrowser:
            st.markdown(f"Proposed insurance premium class: {premium[-1]}")

    return premium
            
insurance_class(preds, True)


################################################         
#plots
#stroke depending on age

def preparing_data_for_plot(x):
    cols = x.columns
    frame = pd.DataFrame(columns = cols)
    for i in range (0, 83):
        x['age'] = i
        frame = frame.append(x)
    frame.set_index(pd.Index(range(0, 83)))
    return frame


if st.checkbox("Show stroke risk assessment", False):
    df2 = preparing_data_for_plot(df)
    preds_age = model.predict(df2)

    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df2['age'], preds_age, label = 'Stroke Risk Prediction')
    ax1.axvline(age, color = 'green', label = 'Current Age')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Stroke Risk')
    ax1.set_title('Stroke Risk in relation to age')
    ax1.legend()
    st.pyplot(fig1)

def plot(st, model):
    import matplotlib.pyplot as plt

    # Hard coding labels is not a good thing to do normally ...
    labels = {'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Female', 'gender_Male', 'gender_Other', 'ever_married_No',
            'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
            'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Rural', 'Residence_type_Urban',
            'smoking_status_formerly smoked', 'smoking_status_never smoked',
            'smoking_status_smokes', 'smoking_status_unknown'}
    sizes = model.feature_importances_

    assert(len(labels)==len(sizes))

    data = list(zip(sizes, labels))
    data = sorted(data, key= lambda x: x[0], reverse=True)
    data = list(zip(*data))
    sizes, labels = data[0], data[1]

    plt.rcParams.update({
        "figure.facecolor":  (1.0, 0.0, 0.0, 0.3),  # red
        "axes.facecolor":    (0.0, 1.0, 0.0, 0.0),  # green
        "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),  # blue  
    })
    
#########################################
#uploader section

st.header("Insurance Premiums")

uploaded_data = st.file_uploader("Choose a file with Customer Data for suggestions regarding insurance premiums")

def imputing_bmi(dataset):
    male_bmi = data[data['gender'] == 'Male'][['bmi', 'age']].dropna()
    female_bmi = data[data['gender'] == 'Female'][['bmi', 'age']].dropna()
    male_bmi = male_bmi.groupby('age').mean()
    female_bmi = female_bmi.groupby('age').mean()
    total_bmi = data[['age', 'bmi']].dropna()
    total_bmi = total_bmi.groupby('age').mean()
    
    for i in dataset.loc[(dataset['gender'] == 'Male') & (dataset['bmi'].isnull())].index:
        age = dataset.loc[i]['age']
        bmi = male_bmi.iloc[age]
        dataset.loc[i] = dataset.loc[i].fillna(bmi)
    for i in dataset.loc[(dataset['gender'] == 'Female') & (dataset['bmi'].isnull())].index:
        age = dataset.loc[i]['age']
        bmi = female_bmi.iloc[age]
        dataset.loc[i] = dataset.loc[i].fillna(bmi)
    for i in dataset.loc[(dataset['gender'] == 'Other') & (dataset['bmi'].isnull())].index:
        age = dataset.loc[i]['age']
        bmi = total_bmi.iloc[age]
        dataset.loc[i] = dataset.loc[i].fillna(bmi)
        
    return dataset

@st.cache()
def preparing_data(x):
    x = x.drop('id', axis = 1)
    x['age'] =  x['age'].astype(int)
    x.loc[(x['age'] <= 9, ['smoking_status'])] = x.loc[(x['age'] <= 9, ['smoking_status'])].fillna('never smoked')
    x.loc[(x['age'] > 9 , ['smoking_status'])] = x.loc[(x['age'] > 9, ['smoking_status'])].fillna('unknown')
    x = imputing_bmi(x)
    x = pd.get_dummies(x)
    if 'stroke' in x.columns:    
        x = x.drop('stroke', axis = 1)
   
    return x

@st.cache()
def convert_to_csv(x):
    return x.to_csv(index = False).encode('utf-8')

#add action to be done if file is uploaded


if uploaded_data is not None:
    
    new_customers = pd.read_csv(uploaded_data)
    modified_data = preparing_data(new_customers)
    preds = model.predict(modified_data)

    premium = insurance_class(preds, False)
    #le = LabelEncoder()
    #premium = le.fit_transform(premium.astype(float))
   
    
    new_customers["Insurance premium class"] = premium
    
    
    st.success("🕺🏽🎉👍 You have succesfully assigned %i new customers to their respective insurance premium classes 🕺🏽🎉👍!" % new_customers.shape[0])
    
    download_file = convert_to_csv(new_customers)
    modified_data_file = convert_to_csv(modified_data)
   
    
    row6_col1, row6_col2 = st.columns([1,1])
    
    row6_col1.download_button("Download File",
                       data = download_file,
                       file_name = 'Stroke_Predictions.csv')
    
    row6_col2.download_button("Download modified Data",
                       data = modified_data_file,
                       file_name = 'Modified_Data.csv')
    st.write(new_customers)
    
   
