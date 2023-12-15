import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class_names = os.listdir('./datasets/training_set/')

st.title(":red[Computer Vision]", anchor=False)

tab_intro, tab_cnn, tab_gtm = st.tabs(["Introductie", "Convolutional Neural Network", "Google Teachable Machine"])

def get_eda():
    fig, axes = plt.subplots(len(class_names), 5, figsize=(12, 12))

    for i, class_name in enumerate(class_names):
        class_path = os.path.join('./datasets/training_set/', class_name)

        # get 5 random images from the class
        image_files = os.listdir(class_path)
        random_images = random.sample(image_files, 5)

        # show the images
        for j, image_name in enumerate(random_images):
            image_path = os.path.join(class_path, image_name)
            img = imread(image_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
        
        axes[i, 2].set_title(f'{class_name.replace("_", " ")}: {len(image_files)} training images', fontsize=18, fontweight='bold')
        axes[i, 0].title.set_position([0.5, 5])  # Adjust title position

    # add some padding to the rows
    plt.tight_layout(h_pad=5)

    st.pyplot(fig)

def train_model(epoch_amount):
    loading_text = st.empty()
    loading_text.text("Model trainen...")

    train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                    rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    image_size = 64

    training_set = train_val_datagen.flow_from_directory('datasets/training_set',
                                                    subset='training',
                                                    target_size = (image_size, image_size),
                                                    batch_size = 25,
                                                    class_mode = 'categorical') 

    validation_set = train_val_datagen.flow_from_directory('datasets/training_set',
                                                    subset='validation',
                                                    target_size = (image_size, image_size),
                                                    batch_size = 25,
                                                    class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('datasets/test_set',
                                                target_size = (image_size, image_size),
                                                batch_size = 25,
                                                class_mode = 'categorical')
    
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), input_shape = (image_size, image_size, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(5, activation="softmax")
    ])

    # Compile and train the model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    history = model.fit(training_set,
                    validation_data = validation_set,
                    epochs = epoch_amount,
                    )
    
    loading_text.text("")

    # st.write(f"<p style='font-size: 20px; display: inline; margin-bottom: 5px;'>De loss en accuracy over de epochs: </p>", unsafe_allow_html=True)
    st.subheader(f"De loss en accuracy over de epochs:")
    #plotting the loss and accuracy graphs
    # Create a figure and a grid of subplots with a single call
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # Plot the loss curves on the first subplot
    ax1.plot(history.history['loss'], label='training loss', color='blue')
    ax1.plot(history.history['val_loss'], label='validation loss', color='red')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(history.history['accuracy'], label='training accuracy', color='blue')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy', color='red')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    st.pyplot(fig)
    
    # st.write(f"<p style='font-size: 20px; display: inline; margin-bottom: 5px;'>Evaluatie van het model: </p>", unsafe_allow_html=True)
    st.subheader(f"Evaluatie van het model:")

    true_labels = []
    predicted_labels = []

    # of each batch in the test set, get the actual labels and the predicted labels for their images
    for i in range(len(test_set)):
        images, labels = test_set[i]
        true_labels.extend(np.argmax(labels, axis=1))
        predicted_labels.extend(np.argmax(model.predict(images), axis=1))

    st.write('De actuele labels en voorspelde labels van een paar images:')
    st.write(str(true_labels[:40]))
    st.write(str(predicted_labels[:40]))

    st.write(f"<p style='font-size: 18px; display: inline'>Accuracy: </h3> <p style='display: inline; font-size: 18px; color: #FF4B4B;'>{accuracy_score(true_labels, predicted_labels)}</p>", unsafe_allow_html=True)

    st.subheader("Confusion matrix:")

    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plot.plot(cmap='Reds', ax=ax, xticks_rotation=45, colorbar=False)
    plt.xlabel('Predicted', fontweight='bold', fontsize=12)
    plt.ylabel('Actual', fontweight='bold', fontsize=12)
    plt.title('Confusion Matrix', fontweight='bold', fontsize=18)

    # Display the plot in Streamlit
    st.pyplot(fig)


with tab_intro:
    st.header("Introductie")

    st.markdown("""
                Het thema van mijn computer vision opdracht is een safari in Afrika.

                Ik heb gekozen voor 5 dieren die elk wat unieke kenmerken hebben zodat ze onderscheidbaar zijn van elkaar.
    """)

    get_eda()


with tab_cnn:
    st.header("Convolutional Neural Network")

    st.markdown("""
                Train het (gelimiteerde) model.
                
                Deze versie van het CNN is minder zwaar dan hetgene in mijn notebook, zodat streamlit het aankan.
            
                Het aantal epochs dat je kan kiezen is gelimiteerd, en de image size is 64x64 in plaats van 128x128.
                """)
    epoch_amount = st.slider("Aantal epochs", min_value=3, max_value=15)

    if st.button('Train the model!'):
        train_model(epoch_amount)

with tab_gtm:
    st.header("Google Teachable Machine")
    
    st.subheader("De classes met training data")
    st.image('./images/gtm_classes.png', use_column_width=True)

    st.subheader("Het model uittesten")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.image('./images/gtm_tryout.png', use_column_width=True) 
    with c2:
        st.image('./images/gtm_tryout2.png', use_column_width=True) 
    with c3:
        st.image('./images/gtm_tryout3.png', use_column_width=True) 

    st.subheader("Confusion matrix")
    st.image('./images/gtm_cmpng.png', use_column_width=False)


