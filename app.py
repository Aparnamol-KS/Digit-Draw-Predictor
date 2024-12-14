import gradio as gr
import tensorflow as tf


model = tf.keras.models.load_model('model.h5')

def recognize_digit(image):
    if image is not None:
        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255
        prediction = model.predict(image)
        
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return ''
    
custom_css = """
body {
    font-family: 'Roboto', sans-serif;
    background: linear-gradient(to bottom right, #f0f4f8, #d9e4f5);
    margin: 0;
    padding: 0;
}
#interface-title {
    text-align: center;
    color: #333;
    font-size: 2.5em;
    margin-bottom: 10px;
}
#interface-description {
    text-align: center;
    color: #555;
    margin-bottom: 20px;
    font-size: 1.2em;
}
button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 1em;
    cursor: pointer;
}
button:hover {
    background-color: #45a049;
}
"""

# Define the Gradio interface
interface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Image(
        shape=(28, 28), 
        image_mode="L", 
        invert_colors=True, 
        source="canvas", 
        label="Draw Your Digit Here"
    ),
    outputs = gr.Label(num_top_classes=10),
    live=True,
    title="DigitGenie",
    description="Draw a number on the canvas, and let the model predict it!",
    css=custom_css
)


interface.launch()
