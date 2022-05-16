import requests, json
from bs4 import BeautifulSoup

def retrieve_mc_pron(character):
    url = "https://en.wiktionary.org/w/api.php"
    
    # test with a character with multiple pronunciations
    # character = '不'
    options = {
        'action': 'parse',
        'page': 'Module:zh/data/ltc-pron/'+character,
        'format': 'json'
    }
    mc_descriptions = []
    response_text = requests.post(url, options).text
    if 'parse' in response_text:
        html = json.loads(response_text)['parse']['text']['*']
        soup = BeautifulSoup(html, "html.parser")
        # print(character)

        for l in soup.find_all('span', {'class': 's2'}):
            # print('\t'+l.text.replace('"', ''))
            mc_descriptions.append(l.text)
    else:
        print(f"Nothing found for {character}")
    return mc_descriptions

