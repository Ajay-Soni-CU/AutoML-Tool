from shiny import App, ui, render, reactive
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = None
X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
model = None

def missing_val_impute(value):
    global data
    if value == 1:
        data = data.dropna()
    elif value == 2:
        for i in data.columns:
            if data[i].dtype != "O":  # Only fill numeric columns with mean
                data[i] = data[i].fillna(data[i].mean())
    elif value == 3:
        for i in data.columns:
            if data[i].dtype != "O":  # Only fill numeric columns with median
                data[i] = data[i].fillna(data[i].median())

def data_type_change(column_name, choice):
    global data
    if choice == 1:
        data[column_name] = data[column_name].astype('int')
    elif choice == 2:
        data[column_name] = data[column_name].astype('float')

def encoding_fn(column_name, method):
    global data
    if method == "One Hot Encoding" and data[column_name].nunique() <= 10:
        en = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = en.fit_transform(data[[column_name]])
        for i in range(encoded_data.shape[1]):
            data[f"En_{column_name}_{i}"] = encoded_data[:, i]
    elif method == "Label Encoding" and data[column_name].nunique() <= 20:
        en = LabelEncoder()
        data[f"En_{column_name}"] = en.fit_transform(data[column_name])

def polynomial_features(selected):
    global data
    pf = PolynomialFeatures(degree=2, include_bias=False)
    if len(selected) == 1:
        selected = selected[0]
        poly_data = pf.fit_transform(data[[selected]])
    else:
        selected = [col[0] if isinstance(col, tuple) else col for col in selected]
        poly_data = pf.fit_transform(data[selected])
    for i in range(poly_data.shape[1]):
        data[f"Poly_{i}"] = poly_data[:, i]

def feature_selections(feature, target):
    global data, X, y, X_train, X_test, y_train, y_test
    feature = [col[0] if isinstance(col, tuple) else col for col in feature]
    X = data[feature]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model_compare_R():
    global X_train, X_test, y_train, y_test
    model1 = LinearRegression()
    model2 = RandomForestRegressor(random_state=42)
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    score1 = model1.score(X_test, y_test)
    score2 = model2.score(X_test, y_test)
    return score1, score2

def model_compare_C():
    global X_train, X_test, y_train, y_test
    model1 = LogisticRegression(random_state=42)
    model2 = SVC(kernel='linear', C=1)
    model3 = RandomForestClassifier(random_state=42)
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    score1 = model1.score(X_test, y_test)
    score2 = model2.score(X_test, y_test)
    score3 = model3.score(X_test, y_test)
    return score1, score2, score3

def model_choose(selection):
    global model
    if selection == 'Linear Regression':
        model = LinearRegression()
    elif selection == 'Logistic Regression':
        model = LogisticRegression(random_state=42)
    elif selection == 'Random Forest Regressor':
        model = RandomForestRegressor(random_state=42)
    elif selection == 'Support Vector Classifier':
        model = SVC(kernel='linear', C=1)
    elif selection == 'Random Forest Classifier':
        model = RandomForestClassifier(random_state=42)
    
    model.fit(X_train, y_train)

def predict(pred_features):
    predicted = model.predict(pred_features)
    return predicted

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
             * {
    font-family: 'Space Grotesk', sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    min-height: 100vh;
    overflow-x: hidden;
    color: #2d3748;
    line-height: 1.6;
}

.main-wrapper {
    display: flex;
    min-height: 100vh;
}

.sidebar-container {
    width: 320px;
    background: #1a202c;
    position: fixed;
    height: 100vh;
    overflow-y: auto;
    z-index: 1000;
    box-shadow: 4px 0 20px rgba(0,0,0,0.1);
    border-right: 1px solid #2d3748;
}

.content-container {
    margin-left: 320px;
    flex: 1;
    min-height: 100vh;
    overflow-y: auto;
    background: #f7fafc;
}

.title-panel {
    background: #2d3748;
    color: white;
    padding: 28px 24px;
    text-align: center;
    border-bottom: 1px solid #4a5568;
}

.title-panel h2 {
    font-weight: 700;
    font-size: 2.4em;
    margin: 0;
    letter-spacing: -0.5px;
    color: #ffffff;
}

.title-panel h4 {
    font-weight: 400;
    margin: 12px 0 0 0;
    opacity: 0.9;
    font-size: 1.1em;
    color: #cbd5e0;
}

.sidebar-content {
    padding: 24px 20px;
}

.sidebar-section {
    background: #2d3748;
    padding: 20px;
    border-radius: 8px;
    margin: 16px 0;
    border: 1px solid #4a5568;
    transition: all 0.2s ease;
}

.sidebar-section:hover {
    border-color: #718096;
    transform: translateY(-1px);
}

.sidebar-section h4 {
    color: #e2e8f0;
    margin-bottom: 18px;
    font-weight: 600;
    padding-bottom: 10px;
    font-size: 1.1em;
    letter-spacing: 0.2px;
    border-bottom: 1px solid #4a5568;
}

.btn-custom {
    width: 100%;
    padding: 12px 16px;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    transition: all 0.2s ease;
    margin: 8px 0;
    font-size: 14px;
    cursor: pointer;
    letter-spacing: 0.3px;
    background: #4a5568;
    color: white;
}

.btn-primary {
        color: white;
}

.btn-success {
    background: #38a169;
    color: white;
}

.btn-warning {
    
    color: white;
}

.btn-danger {
    background: #e53e3e;
    color: white;
}

.btn-info {
    background: #319795;
    color: white;
}

.btn-custom:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    filter: brightness(105%);
}

.main-content {
    padding: 28px;
    background: #f7fafc;
    min-height: 100vh;
}

.section {
    background: white;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
    transition: all 0.2s ease;
}

.section:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.section h4 {
    color: #2d3748;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 12px;
    margin: 0 0 18px 0;
    font-weight: 600;
    font-size: 1.3em;
}

.input-group {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 20px;
    margin: 16px 0;
}

.result-box {
    background: #38a169;
    color: white;
    padding: 16px;
    border-radius: 6px;
    margin: 16px 0;
    text-align: center;
    font-weight: 600;
    font-size: 1em;
    border: 1px solid #2f855a;
}

.model-score {
    background: #3182ce;
    color: white;
    padding: 16px;
    border-radius: 6px;
    margin: 12px 0;
    text-align: center;
    font-weight: 600;
    font-size: 1em;
    border: 1px solid #2b6cb0;
}

.warning-box {
    background: #dd6b20;
    color: white;
    padding: 16px;
    border-radius: 6px;
    margin: 16px 0;
    text-align: center;
    font-weight: 600;
    font-size: 1em;
    border: 1px solid #c05621;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

th {
    background: #4a5568;
    color: white;
    padding: 14px;
    text-align: center;
    font-weight: 600;
    font-size: 1em;
    border-bottom: 1px solid #718096;
}

td {
    padding: 12px;
    text-align: center;
    border-bottom: 1px solid #e2e8f0;
    font-weight: 500;
    background: white;
}

tr:hover {
    background-color: #f7fafc;
}

.file-input {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid #4a5568;
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
    color: #e2e8f0;
    width: 100%;
}

.numeric-input {
    background: white;
    border: 1px solid #cbd5e0;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 6px 0;
    width: 100%;
    font-size: 14px;
    font-family: 'Space Grotesk', sans-serif;
}

.select-input {
    background: white;
    border: 1px solid #cbd5e0;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 6px 0;
    width: 100%;
    font-size: 14px;
    font-family: 'Space Grotesk', sans-serif;
}

.input-group p {
    margin-bottom: 12px;
    color: #4a5568;
    font-weight: 500;
}

.sidebar-section .btn-custom {
    display: flex;
    align-items: center;
    gap: 8px;
    justify-content: flex-start;
    padding-left: 20px;
}

.sidebar-section .btn-custom::before {
    font-size: 1.2em;
}
        """)
    ),
    
    ui.div(
        {"class": "main-container"},
        ui.div(
            {"class": "title-panel"},
            ui.h2("🤖 AUTO ML TOOL"),
            ui.h4("Automated Machine Learning Platform")
        ),
        
        ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    {"class": "sidebar-section"},
                    ui.h4("📁 DATA UPLOAD"),
                    ui.input_file("file", "Upload CSV File", accept=[".csv"])
                ),
                
                ui.div(
                    {"class": "sidebar-section"},
                    ui.h4("🔧 PREPROCESSING"),
                    ui.input_action_button("imputer", "🧹 Impute Missing Values", class_="btn-custom btn-primary"),
                    ui.input_action_button("dtype", "🔄 Change Data Types", class_="btn-custom btn-warning"),
                    ui.input_action_button("en", "🎯 Feature Encoding", class_="btn-custom btn-success"),
                    ui.input_action_button("poly", "📊 Polynomial Features", class_="btn-custom btn-info")
                ),
                
                ui.div(
                    {"class": "sidebar-section"},
                    ui.h4("🤖 MODEL BUILDING"),
                    ui.input_action_button("target", "🎯 Select Features & Target", class_="btn-custom btn-primary"),
                    ui.input_action_button("comp", "📈 Compare Models", class_="btn-custom btn-warning"),
                    ui.input_action_button("select", "⚡ Select & Train Model", class_="btn-custom btn-success"),
                    ui.input_action_button("pred", "🔮 Make Predictions", class_="btn-custom btn-danger")
                ),
                width=350,
            ),
            
            ui.div(
                {"class": "main-content"},
                ui.output_table("data_input"),
                ui.output_ui("imputer_ui"),
                ui.output_ui("imputing"),
                ui.output_ui("dtype_change"),
                ui.output_ui("dtyping"),
                ui.output_ui("encoder_method"),
                ui.output_ui("encoding"),
                ui.output_ui("polynomial_method"),
                ui.output_ui("polynoming"),
                ui.output_ui("feature_selection"),
                ui.output_ui("featuring"),
                ui.output_ui("compare"),
                ui.output_ui("model_selection"),
                ui.output_ui("modeling"),
                ui.output_ui("predict_inputs"),
                ui.output_ui("predict_inputs2"),
                ui.output_ui("predicting"),
            )
        )
    )
)

def server(input, output, session):
    @output
    @render.table
    def data_input():
        global data
        path = input.file()
        if not path:
            return pd.DataFrame()
        data = pd.read_csv(path[0]["datapath"])
        return data.head()

    @output
    @render.ui
    @reactive.event(input.imputer)
    def imputer_ui():
        return ui.div(
            {"class": "section"},
            ui.h4("🧹 Missing Value Imputation"),
            ui.div(
                {"class": "input-group"},
                ui.p("Choose imputation method:"),
                ui.input_numeric("choice", "1: Drop NA, 2: Fill Mean, 3: Fill Median", value=1, min=1, max=3),
                ui.input_action_button("start_imp", "Apply Imputation", class_="btn-custom btn-primary")
            )
        )

    @output
    @render.text
    @reactive.event(input.start_imp)
    def imputing():
        choice_val = input.choice()
        missing_val_impute(choice_val)
        return ui.div({"class": "result-box"}, "✅ Missing Values Imputed Successfully")

    @output
    @render.ui
    @reactive.event(input.dtype)
    def dtype_change():
        return ui.div(
            {"class": "section"},
            ui.h4("🔄 Data Type Conversion"),
            ui.div(
                {"class": "input-group"},
                ui.p("Convert feature data type:"),
                ui.input_numeric("choice2", "1: To Integer, 2: To Float", value=1, min=1, max=2),
                ui.input_select("s_feature", "Select Feature:", choices=list(data.columns)),
                ui.input_action_button("start_dtype", "Convert Data Type", class_="btn-custom btn-warning")
            )
        )

    @output
    @render.text
    @reactive.event(input.start_dtype)
    def dtyping():
        s_val = input.s_feature()
        choice_val = input.choice2()
        data_type_change(s_val, choice_val)
        return ui.div({"class": "result-box"}, f"✅ Changed Data Type of Feature: {s_val}!")

    @output
    @render.ui
    @reactive.event(input.en)
    def encoder_method():
        return ui.div(
            {"class": "section"},
            ui.h4("🎯 Feature Encoding"),
            ui.div(
                {"class": "input-group"},
                ui.p("Encode categorical features:"),
                ui.input_select("s_feature2", "Encoding Method:", choices=["One Hot Encoding", "Label Encoding"]),
                ui.input_select("s_feature3", "Select Feature:", choices=list(data.columns)),
                ui.input_action_button("start_en", "Apply Encoding", class_="btn-custom btn-success")
            )
        )

    @output
    @render.text
    @reactive.event(input.start_en)
    def encoding():
        encoding_fn(input.s_feature3(), input.s_feature2())
        e_val = input.s_feature3()
        return ui.div({"class": "result-box"}, f"✅ Encoding Complete: {e_val}")

    @output
    @render.ui
    @reactive.event(input.poly)
    def polynomial_method():
        return ui.div(
            {"class": "section"},
            ui.h4("📊 Polynomial Features"),
            ui.div(
                {"class": "input-group"},
                ui.p("Create polynomial features:"),
                ui.input_select("s_feature5", "Select Features:", choices=list(data.columns), multiple=True),
                ui.input_action_button("start_poly", "Create Features", class_="btn-custom btn-info")
            )
        )

    @output
    @render.text
    @reactive.event(input.start_poly)
    def polynoming():
        polynomial_features(input.s_feature5())
        return ui.div({"class": "result-box"}, "✅ Polynomial Features Created!")

    @output
    @render.ui
    @reactive.event(input.target)
    def feature_selection():
        return ui.div(
            {"class": "section"},
            ui.h4("🎯 Feature & Target Selection"),
            ui.div(
                {"class": "input-group"},
                ui.p("Select features and target for modeling:"),
                ui.input_select("s_feature6", "Features (multiple):", choices=list(data.columns), multiple=True),
                ui.input_select("s_feature7", "Target Variable:", choices=list(data.columns)),
                ui.input_action_button("start_feat", "Confirm Selection", class_="btn-custom btn-primary")
            )
        )

    @output
    @render.text
    @reactive.event(input.start_feat)
    def featuring():
        feature_selections(input.s_feature6(), input.s_feature7())
        return ui.div({"class": "result-box"}, "✅ Features and Target are Selected")

    @output
    @render.ui
    @reactive.event(input.comp)
    def compare():
        global y_test
        if y_test is None:
            return ui.div(
                {"class": "section"},
                ui.h4("📈 Model Comparison"),
                ui.div({"class": "warning-box"}, "⚠️ Please select features and target first!")
            )
        
        unique = y_test.nunique()
        if (y_test.dtype in ["float64", "float32"]) and unique > 20:
            as1, as3 = model_compare_R()
            return ui.div(
                {"class": "section"},
                ui.h4("📈 Regression Model Comparison"),
                ui.div({"class": "model-score"}, f"Linear Regression: {as1:.4f}"),
                ui.div({"class": "model-score"}, f"Random Forest Regressor: {as3:.4f}")
            )
        else:
            as2, as4, as5 = model_compare_C()
            return ui.div(
                {"class": "section"},
                ui.h4("📊 Classification Model Comparison"),
                ui.div({"class": "model-score"}, f"Logistic Regression: {as2:.4f}"),
                ui.div({"class": "model-score"}, f"Support Vector Classifier: {as4:.4f}"),
                ui.div({"class": "model-score"}, f"Random Forest Classifier: {as5:.4f}")
            )

    @output
    @render.ui
    @reactive.event(input.select)
    def model_selection():
        global y_test
        if y_test is None:
            return ui.div(
                {"class": "section"},
                ui.h4("⚡ Model Selection"),
                ui.div({"class": "warning-box"}, "⚠️ Please select features and target first!")
            )
        
        unique = y_test.nunique()
        if (y_test.dtype in ["float64", "float32"]) and unique > 20:
            ls = ['Linear Regression', 'Random Forest Regressor']
        else:
            ls = ['Logistic Regression', 'Support Vector Classifier', 'Random Forest Classifier']
        
        return ui.div(
            {"class": "section"},
            ui.h4("⚡ Model Selection & Training"),
            ui.div(
                {"class": "input-group"},
                ui.p("Choose your model:"),
                ui.input_select("s_feature8", "Select Model:", choices=ls),
                ui.input_action_button("start_select", "Train Model", class_="btn-custom btn-success")
            )
        )

    @output
    @render.text
    @reactive.event(input.start_select)
    def modeling():
        model_choose(input.s_feature8())
        return ui.div({"class": "result-box"}, f"✅ {input.s_feature8()} trained successfully!")

    @output
    @render.ui
    @reactive.event(input.pred)
    def predict_inputs():
        global X_test
        if X_test is not None:
            return ui.div(
                {"class": "section"},
                ui.h4("🔮 Make Predictions"),
                ui.div(
                    {"class": "input-group"},
                    ui.p("Enter feature values for prediction:"),
                    *[ui.input_numeric(feature, f"📊 {feature}:", value=0) for feature in X_test.columns]
                )
            )
        else:
            return ui.div(
                {"class": "section"},
                ui.h4("🔮 Make Predictions"),
                ui.div({"class": "warning-box"}, "⚠️ Please select features and target first!")
            )

    @output
    @render.ui
    @reactive.event(input.pred)
    def predict_inputs2():
        return ui.input_action_button("start_pred", "🚀 Predict", class_="btn-custom btn-danger")

    @output
    @render.text
    @reactive.event(input.start_pred)
    def predicting():
        global X_test
        if X_test is None:
            return ui.div({"class": "warning-box"}, "⚠️ Please select features and target first!")
        
        feature_values = [input[feature]() for feature in X_test.columns]
        predicted = predict([feature_values])
        return ui.div({"class": "result-box"}, f"🎯 Prediction Result: {predicted[0]}")

app = App(app_ui, server)