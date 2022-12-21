import tensorflow as tf
import numpy as np


def predict(image_name):
    class_names = ['Your Potaoes are infected by <b>Early Blight Disease</b>',
                   'Your Potatoes are infected by <b>Late Blight Disease</b>', 'Your potatoes are <b>healthy</b>']

    model = tf.keras.models.load_model('potatoes.h5')
    # Load the image
    image = tf.io.read_file(image_name)

    # Decode the image
    image = tf.image.decode_image(image)

    # Preprocess the image
    image = tf.image.resize(image, (256, 256))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values

    # Add an additional dimension to the input tensor
    image = tf.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)
    disease = class_names[np.argmax(predictions)]
    if 'early' in disease:
        remedy = "💡 To treat this disease, it is important to follow these steps:" \
                 "\n\n✅ Remove and destroy infected plants: Remove any infected plants from the field and destroy them to prevent the spread of the disease." \
                 "\n\n✅ Apply a fungicide: There are several fungicides available that can be used to control early blight. Follow the instructions on the product label for the recommended application rate and frequency." \
                 "\n\n✅ Practice proper cultural control measures: To prevent the spread of early blight, it is important to practice good cultural control measures, such as rotating crops, maintaining proper plant spacing, and watering at the base of the plant to prevent the spread of the disease." \
                 "\n\n✅ Use resistant varieties: Planting potato varieties that are resistant to early blight can help reduce the risk of the disease." \
                 "\n\n✅ Avoid planting in infected soil: If the soil is infected with early blight, avoid planting potatoes in that soil to prevent the disease from spreading." \
                 "\n\nBy following these steps, you can effectively treat and prevent potato early blight disease."

    elif 'late' in disease:
        remedy = "💡To treat late blight, there are several steps you can take:" \
                 "\n\n✅Remove and destroy infected plants: Remove and destroy any infected plants, including the tubers, to prevent the disease from spreading. Be sure to properly dispose of the plants, as they can continue to harbor the fungus and spread the disease." \
                 "\n\n✅Use fungicides: Fungicides, such as copper-based products, can be applied to help prevent the spread of late blight. Follow the instructions on the label for proper application." \
                 "\n\n✅Practice good cultural practices: Avoid overhead watering, as this can help reduce the spread of the fungus. Instead, water the plants at the base to keep the leaves dry. Rotate your crops to avoid planting potatoes in the same location year after year, as this can lead to a buildup of the fungus in the soil." \
                 "\n\n✅Plant resistant varieties: Some potato varieties are more resistant to late blight than others. Consider planting these varieties in your garden to help reduce the risk of the disease." \
                 "\n\n✅Monitor your plants regularly: Regularly check your potato plants for signs of late blight and take action if you notice any symptoms." \
                 "\n\nBy following these steps, you can help prevent or manage late blight in your potato plants."
    else:
        remedy = "💡There are several steps you can take to help keep your potato crops healthy:" \
                 "\n\n✅Choose the right location: Plant your potatoes in a sunny location with well-draining soil. Avoid areas that are prone to standing water, as this can lead to rot." \
                 "\n\n✅Use the right soil: Potatoes grow best in soil that is rich in organic matter and has a pH between 5.0 and 7.0. Adding compost or well-rotted manure to the soil can help improve its structure and fertility." \
                 "\n\n✅Plant at the right time: In most areas, potatoes are planted in the spring after the last frost date. Be sure to check the specific planting recommendations for your region." \
                 "\n\n✅Use the proper spacing: Proper spacing is important for healthy potato plants. Follow the specific spacing recommendations for the variety you are planting, as different varieties have different spacing requirements." \
                 "\n\n✅Water consistently: Potatoes need consistent watering to grow properly. Water your plants deeply and regularly, providing about 1 inch of water per week. Avoid overhead watering, as this can encourage fungal growth." \
                 "\n\n✅Fertilize as needed: Potatoes are heavy feeders and may benefit from fertilization. Use a balanced fertilizer or add compost to the soil to provide the nutrients your plants need." \
                 "\n\n✅Practice good pest and disease control: Monitor your plants regularly for signs of pests or disease and take appropriate action if necessary. This may include using pest controls or removing and destroying infected plants." \
                 "\n\nBy following these steps, you can help ensure that your potato crops are healthy and productive."
    return [disease, remedy]
