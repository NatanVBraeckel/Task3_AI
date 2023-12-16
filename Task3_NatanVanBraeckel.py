import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class_names = os.listdir('./datasets/training_set/')

st.title(":red[Computer Vision]", anchor=False)

tab_intro, tab_cnn, tab_gtm, tab_scraper = st.tabs(["Introductie", "Convolutional Neural Network", "Google Teachable Machine", "Image Scraper"])

def get_eda():
    st.subheader("Exploratory Data Analysis")
    st.markdown("""
                Een paar willekeurig geselecteerde foto's voor elk dier.
                """)

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
        
        axes[i, 2].set_title(f'{class_name.replace("_", " ")}: {len(image_files)} training images', fontsize=18, fontweight='bold', pad=10)
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

    st.subheader("Een paar images van de test set uitproberen:")
    # get a random batch index from the test set
    random_batch_idx = random.randint(0, len(test_set) - 1)

    # get the batch (2d list of images and labels)
    batch = test_set[random_batch_idx]

    # generate a list of random 5 images/labels indexes from batch
    random_test_indexes = random.sample(range(len(batch[0])), 5)

    for idx in random_test_indexes:
        fig, axes = plt.subplots(1,1, figsize=(12, 12))

        images, labels = batch
        
        image = images[idx]
        label = labels[idx]
        #label is e.g. [0. 0. 0. 1. 0.]
    
        actual_label = np.where(label == 1)[0][0]

        prediction = model.predict(np.expand_dims(image, axis=0))
        predicted_label = np.argmax(prediction, axis=1)

        # check if actual and predicted labels match
        label_color = 'green' if actual_label == predicted_label else 'maroon'

        plt.imshow(image)
        plt.title(f"Actual: {class_names[actual_label]}\nPredicted: {class_names[predicted_label[0]]}", color=label_color)
        plt.axis('off')
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

with tab_scraper:
    st.header("Image Scraper")

    st.write(":orange[DEZE PAGINA IS NA DE DEADLINE TOEGEVOEGD.]")
    st.write(":yellow[De scraper had ik niet bij in mijn notebook gezet, zodat deze niet altijd geactiveerd zou worden bij het runnen van de hele notebook.]")
    st.write(":yellow[Ik ben die dan uiteindelijk vergeten er bij in te zetten. Maar ik wou mijn code toch nog laten zien, vandaar deze pagina in de streamlit app.]")

    st.subheader("Alle code")
    st.image('./images/scraper/scraper_all_code.png')

    st.header("De functie", anchor=False)
    st.subheader("Image sources scrapen")
    st.markdown("""
                Eerst wordt de browser opgestart en gesurft naar Flickr met de zoekterm die de gebruiker heeft ingegeven.

                Daarna wordt er een lus opgestart. Zolang er niet genoeg images zijn zal er nog wat verder gescrollt worden om meer images in te laden.

                Als er genoeg images gevonden zijn, worden alle src attributen (de link naar de image) opgehaald en opgeslagen in photo_sources.

                Die array wordt ook gesliced, zodat er evenveel images zijn als gevraagd.

                De driver/browser kan dan ook afgesloten worden.
                """)
    st.image('./images/scraper/scraper_getting_images.png')

    st.subheader("Images downloaden")
    st.markdown("""
                Nudat alle links naar de images zijn opgehaald, kunnen ze 1 voor 1 gedownload worden.

                Er worden folders aangemaakt in de training_set en test_set folder met de zoekterm als naam.

                (b.v. /datasets/training_set/elephant en /datasets/test_set/elphant)

                Dan worden de images 1 voor 1 gedownload.

                De index van de image wordt vergeleken met het aantal foto's dat de gebruiker heeft opgevraagd.
                Als deze kleiner is dan 80% van het aantal gevraagde foto's zal de foto in de training set geplaatsd worden, anders in de test set.

                Zo zullen de foto's automatisch verdeeld worden in 80% training set / 20% test set (b.v. 250 images => 200 training + 50 test images).
                """)
    st.image('./images/scraper/scraper_downloading_images.png')

    st.header("De functie oproepen")
    st.markdown("""
                De functie kan opgeroepen worden door een zoekterm en een aantal gevraagde images mee te geven.

                Dit kan bijvoorbeeld voorgeprogammeerd met een lus, of het kan voor opstarten aan de gebruiker gevraagd worden.
                """)
    st.image('./images/scraper/scraper_loop.png')
    st.image('./images/scraper/scraper_ask_user.png')
