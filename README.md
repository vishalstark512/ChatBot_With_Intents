# ChatBot_With_Intents
Chatbot based on intents


There are 3 files in this repositiry 
"intents.json" file is for holding the chat conversations
"generate_data.py" to train you neural network on the give dataset
And the last "chat_model.py" for creating the responses for the question asked

to use the wikipedia, news, google, and weather, which are working online you need to specify the following words
in your querry wiki, news, google & weather
News is just in raw format right now, it needs to be cleaned up and in weather you have to enter the city after entering
the querry.

***If you want to remove the audio responses

remove line 109-112 in chat_model.py

and do just return(random.choice(i['responses']))

You can always add more data to json file to improve the accuracy
