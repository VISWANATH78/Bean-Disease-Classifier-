import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Bean Disease Classifier",
    page_icon="🌱",
    layout="centered"
)

# Define class names
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']

def train_model():
    """Train the model and return it"""
    # Load the beans dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'beans',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )
    
    # Preprocess the data
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        return image, label
    
    # Create training and test datasets
    BATCH_SIZE = 32
    ds_train = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE)
    ds_test = ds_test.map(preprocess).batch(BATCH_SIZE)
    
    # Create the model using functional API
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Load the MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Train the model
    with st.spinner('Training model... This may take a few minutes.'):
        history = model.fit(
            ds_train,
            validation_data=ds_test,
            epochs=10,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
    
    # Save the model with .h5 extension
    try:
        model.save('bean_classifier.h5')
        st.success('Model training completed and saved!')
    except Exception as e:
        st.error(f'Error saving model: {str(e)}')
    
    return model

def load_model():
    """Load the trained model or train if it doesn't exist"""
    try:
        model = tf.keras.models.load_model('bean_classifier.h5')
        st.success('Loaded existing model!')
        return model
    except Exception as e:
        st.warning(f'No existing model found ({str(e)}). Training new model...')
        return train_model()

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model's expected sizing
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array

def main():
    # Page title and description
    st.title("Bean Disease Classifier 🌱")
    st.write("""
    Upload an image of a bean plant leaf, and I'll help you identify if it's healthy 
    or affected by diseases like angular leaf spot or bean rust.
    """)
    
    # Load or train model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Add a prediction button
            if st.button('Analyze Image'):
                with st.spinner('Analyzing...'):
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predictions = model.predict(processed_image)
                    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                    confidence = float(np.max(predictions[0]))
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Show prediction with confidence
                    st.subheader("Results:")
                    st.write(f"**Prediction:** {predicted_class.replace('_', ' ').title()}")
                    st.write(f"**Confidence:** {confidence:.2%}")
                    
                    # Display all class probabilities
                    st.subheader("Detailed Probabilities:")
                    for class_name, prob in zip(CLASS_NAMES, predictions[0]):
                        st.write(f"{class_name.replace('_', ' ').title()}: {prob:.2%}")
                    
                    # Provide recommendations
                    st.subheader("Recommendations:")
                    if predicted_class == 'healthy':
                        st.write("✅ Your bean plant appears to be healthy! Continue with regular care and monitoring.")
                    elif predicted_class == 'angular_leaf_spot':
                        st.write("""
                        🚨 Angular Leaf Spot detected. Recommended actions:
                        - Remove infected leaves
                        - Improve air circulation
                        - Apply appropriate fungicide
                        - Avoid overhead irrigation
                        """)
                    else:  # bean_rust
                        st.write("""
                        🚨 Bean Rust detected. Recommended actions:
                        - Remove and destroy infected plants
                        - Apply fungicide treatment
                        - Maintain proper plant spacing
                        - Keep leaves dry
                        """)
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading a different image.")
    
    # Add information about the supported diseases
    with st.expander("Learn More About Bean Diseases"):
        st.write("""
        ### Angular Leaf Spot
        A fungal disease characterized by angular-shaped spots on leaves with light brown centers 
        and darker borders.
        
        ### Bean Rust
        A fungal disease that appears as rusty brown spots on leaves, typically circular or oval-shaped.
        
        ### Prevention Tips
        - Practice crop rotation
        - Maintain proper plant spacing
        - Avoid overhead irrigation
        - Use disease-resistant varieties when possible
        """)
    
    # Add spacing before footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Add a clean, professional footer
    footer_html = """
    <style>
        .footer {
            position: relative;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            color: #6c757d;
            text-align: center;
            padding: 10px;
            border-top: 1px solid #dee2e6;
            margin-top: 20px;
        }
        .footer p {
            margin: 0;
            font-size: 14px;
            line-height: 1.5;
        }
    </style>
    <div class="footer">
        <p>
            <strong>Bean Disease Classifier</strong><br>
            Developed by <b>Viswanath</b><br>
            © 2024 All Rights Reserved
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
    
    # Remove any extra spacing at the bottom
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
