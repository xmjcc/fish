import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('picklemodel.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data["model"]


def show_predict_page():
    st.title("Fish Type Prediction By Size")
    st.subheader(f"Durham College Student: Wenping Wang Assignment")

    st.write("""### please key in fish size to predict the fish type""")


    lenth1 = st.slider("lenth1", 0, 100, 30 )
    lenth2 = st.slider("lenth2", 0, 100, 30)
    lenth3 = st.slider("lenth3", 0, 100, 30 )
    height = st.slider("height", 0, 20, 8 )
    width = st.slider("width", 0, 20, 4 )

    ok = st.button("Predict fish type")

    if ok:
        X = np.array([[lenth1, lenth2, lenth3, height, width]])
        X = X.astype(float)
        prediction = model.predict(X)
        fishtype = prediction[0]

        dic = {0: "Bream", 1: "Roach", 2: "Whitefish", 3: "Parkti", 4: "Perch", 5: "Pike", 6: "Smelt"}

        st.subheader(f"The estimate fish type is {dic[fishtype]}")
    # 

show_predict_page()

