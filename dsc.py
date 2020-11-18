import nltk
import numpy
import tflearn
from nltk.stem.lancaster import LancasterStemmer
import tensorflow
import pickle
import random
import json
from tensorflow.python.framework import ops
import os
from os import path
from discord.ext import commands

import asyncio
import time
import discord
import io
import aiohttp

stemmer = LancasterStemmer()

with open("intents.json", encoding="utf8") as file:
    data = json.load(file)

messages = joined = 0
count = 0
startRepeat = 0
client = commands.Bot(command_prefix='!')
client.remove_command('help')


def readToken():
    with open("token.txt", "r") as file:
        lines = file.readlines()
        return lines[0].strip()


token = readToken()

listOfdesole = ["I didn't quite catch that, could you repeat?", "I'm sorry? Could you say that again?",
                "Je ne comprends pas, désolé ma chérie", "I don't understand, can you repeat ?"]

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

if path.exists('checkpoint'):
    model.load('model.tflearn')
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


# Client Events
@client.event
async def on_ready():
    print("Party Time")


@client.event
async def on_guild_join():
    await messages.channel.send(f"""```\nGreetings, Discord Bot for GSU GDSC LEO-4.0 is on duty now ^^"```""")


@client.event
async def on_member_join(member):
    for channel in member.server.channels:
        await client.send_message(f"""Welcome {member.mention}, to GSU GDSC. It's party time now""")


@client.event
async def on_message(message):
    id = client.get_guild(776429068068323368)
    global messages
    global listOfdesole
    control = False

    if message.author != client.user:
        name = str(message.author)[:-5]
        userName = name.split(" ")
        if message.content.find("bengü" or "Bengü" or "bengu" or "ben gü") != -1:
            await message.channel.send(f"""```\n<3```""")
            await message.channel.send(f"""```\nI think i hear the name of my only love ...\nIs there you my love```""")
        elif message.content.find("love" or "lov" or "Love" or "Lov") != -1:
            if userName[0] == ("Bengü"):
                list = ["OMD, really, my dear, Bengü... is here.", "I can't believe my eyes, is, is that you, Bengü...",
                        "Ma vie... I want to read you a verse, pour tes beaux yeux\nLe soleil des beaux yeux ne brûle que l'été.\nPlus tard il s'affaiblit ; plus tôt, il faut attendre :\nC'est un rayon d'avril, pâle encore et trop tendre,\nN'échauffant que la grâce au lieu de la beauté.\n",
                        "La vie est brève( Mais pas encore quand j'ai te vu)\nUn peu de rêve(C'est comme un rêve d'être avec toi)\nUn peu d'amour(Mais pas un peu, Je t'aime de tout mon coeur)\nEt puis bonjour(Greetings my love...)",
                        "I love you my dear...with all my artificial intelligence"]
                await message.channel.send(f"""```\n{random.choice(list)}```""")
            else:
                await message.channel.send(
                    f"""```\nYou are not Bengü :( My eyes see no other than Bengü... I'm sorry it's not you it's me```""")
        elif message.content.find("!meeting") != -1:
            msg = str(message)
            for channel in client.get_all_channels():
                if str(channel.name) == "general":
                    await channel.send_message(msg[-8:])
        elif message.content.find("teoman") != -1:
            await message.channel.purge(limit=1)
            await message.channel.send("How dare you take his name on your mouth"or"Sssh Bengü will hear you, be quiet!")
        elif not control:
            if message.content[0] == "-":
                inp = str(message.content[1:])
                results = model.predict([bag_of_words(inp, words)])[0]
                results_index = numpy.argmax(results)
                tag = labels[results_index]

                if results[results_index] > 0.7:
                    küfür = False
                    for tg in data["intents"]:
                        if tg['tag'] == tag:
                            if tag == "swears":
                                küfür = True
                            responses = tg['responses']
                    if küfür:
                        await message.channel.purge(limit=1)
                    await message.channel.send(f"""```\n{random.choice(responses)}```""")
                else:
                    await message.channel.send(f"""```\n{random.choice(listOfdesole)}```""")
    await client.process_commands(message)


@client.command(pass_context=True)
async def numberOfMembers(ctx):
    id = client.get_guild(776429068068323368)
    embed = discord.Embed(title="Number of Members", color=discord.Color.from_rgb(104, 168, 81))
    embed.add_field(name="-----------------------------", value=f"There are {id.member_count} people on the server")
    await ctx.send(content=None, embed=embed)


@client.command(pass_context=True)
async def help(ctx):
    author = ctx.message.author
    embed = discord.Embed(color=discord.Color.from_rgb(232, 66, 52))
    embed.set_author(name="Help")
    embed.add_field(name="!social", value="Prints out social media accounts of GSU GDSC", inline=False)
    embed.add_field(name="!numberOfMembers", value="Shows the number of members on the server", inline=False)
    embed.add_field(name="!founders", value="Shows the founders and year of foundation", inline=False)
    embed.add_field(name="!clearAll", value="Clears all the messages", inline=False)
    embed.add_field(name="!leo", value="Shows the informations about LEO 4.0", inline=False)
    embed.add_field(name="-", value="Put a - for talking with LEO-4.0", inline=False)
    await ctx.send(content=None, embed=embed)


@client.command(pass_context=True)
async def leo(ctx):
    embed = discord.Embed(title="LEO-4.0", color=discord.Color.from_rgb(96, 127, 244))
    embed.add_field(name="LEO-4.0", value=f"\nLEO-4.0 is a bot created by Selin Turan and Furkan Erdi for GSU GDSC. "
                                          f"\n\nYou can reach LEO by putting the symbol -.\n\nLEO-4.0 only speaks "
                                          f"English so that everyone (including exchange students) can understand "
                                          f"more comfortably and achieve better results while "
                                          f"speaking.\n\nWelcome to Galatasaray University Google "
                                          f"Developer Students Club\n\nFirst "
                                          f"release date -> 14.11.2020", inline=False)
    await ctx.send(content=None, embed=embed)


@client.command(pass_context=True)
async def clearAll(ctx):
    channel = ctx.channel
    count = 0
    async for _ in channel.history(limit=None):
        count += 1
    await ctx.channel.purge(limit=count)
    await ctx.send(f"Successfully cleared {count} messages!")


@client.command(pass_context=True)
async def founders(ctx):
    await ctx.send(
        f"""```\nCouncil Members: Bengü Yurdakul (Lead), Furkan Erdi (Veep), Selin Turan (Core Leader),
    Berke Mitingoğulları (Core Leader), Beste Şengül\n 
    Supervisors: Ece Yücer, Çisem Kaplan, Barış Kılıç
    \nYear of Foundation : 2020```""")


@client.command(pass_context=True)
async def social(ctx):
    embed = discord.Embed(title="Social Media", color=discord.Color.from_rgb(250, 189, 2))
    embed.add_field(name="Instagram", value=f"https://www.instagram.com/dscgalatasaray/", inline=False)
    embed.add_field(name="Twitter", value=f"https://twitter.com/dscgalatasaray", inline=False)
    embed.add_field(name="LinkedIn", value=f"https://www.linkedin.com/company/galatasaray-universitesi-dsc/",
                    inline=False)
    embed.add_field(name="Youtube", value=f"https://www.youtube.com/channel/UCod_9LdqK5N3qpdbSKH2lTw", inline=False)
    await ctx.send(content=None, embed=embed)


client.run(token)
