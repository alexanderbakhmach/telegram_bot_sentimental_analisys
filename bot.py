from telegram.ext import Updater, InlineQueryHandler, MessageHandler, Filters
import requests
import re
import tensorflow as tf
from tn.network import TweetAnalysisNetwork
import logging


def print_errors(errors: list):
    """Prints pretty errors

    Args:
        errors (list): A list of error strings
    Return
        None
    """
    if len(errors) > 0:
        for error in errors:
            print('\n')
            print('### Errors')
            print(f'\t- {error}', end='\n\n\n')
            exit()

# Load configs
with open('config.json', 'r') as f:

    # Receive data
    config = json.load(f)
    model_path = config['model_path']
    dataset_path = config['data_path']
    telegram_token = config['telegram_token']

    # Define error buffer
    errors = []
    if not model_path:
        errors.append('The is no model path in config.json file')
    if not dataset_path:
        errors.append('There is no dataset path in config.json file')
    if not telegram_token:
        errors.append('There is no telegram token in config.json file')

    # If there are some errors print them and exit
    print_errors(errors)

# Set logging
logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create new neural model instance to load trained .h5
model = TweetAnalysisNetwork(dataset_path)

# Load trained model
model.load(model_path)

# Receive default tensorflow graph for computetions
graph = tf.get_default_graph()

chart_url = 'https://image-charts.com/chart'\
        '?cht=bvg&chs=600x600&chco=8c592c|6a41b5|369b6a'

def handle_message(bot, update):

    # Receive user chat id
    chat_id = update.message.chat_id

    # Receive user message
    text = update.message.text

    # Under active tf graph make predictions
    with graph.as_default():

        # Make prediction
        prediction = model.predict(text, True)

        # Parse predicted data
        p_neutral, neutral_name = prediction.get('neutral').get('probability'),\
                prediction.get('neutral').get('name')
        p_positive, positive_name = prediction.get('positive').get('probability'),\
                prediction.get('positive').get('name')
        p_negative, negative_name = prediction.get('negative').get('probability'),\
                prediction.get('negative').get('name')

        # Create url to receive chart based on predicted data
        url = f'{chart_url}&chd=t:{p_neutral},{p_positive},{p_negative}'\
                f'&chl={neutral_name}|{positive_name}|{negative_name}'

        # Create information which will showed to user
        information = f'По результатам анализа \n {negative_name} : {p_negative}'\
                f' \n {positive_name}: {p_positive}'\
                f' \n {neutral_name}: {p_neutral}'

        # Send chart to user
        bot.send_photo(chat_id=chat_id, photo=url)

        # Send info to user
        bot.send_message(chat_id=chat_id, text=information)


def main():
    """Enter function
    """
    # Define telegram bot update handler
    updater = Updater(telegram_token)

    # Receive update dispatcher
    dp = updater.dispatcher

    # Set message handle callback to analize messages
    dp.add_handler(MessageHandler(Filters.text, handle_message))

    # Start handling
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
