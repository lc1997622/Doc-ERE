import requests
import json

DEFAULT_PARAMS = {
  "model": "text-davinci-003",
  "temperature": 0,
  "max_tokens": 2048,
  "top_p": 1,
  "frequency_penalty": 0,
  "presence_penalty": 0,
}


param = DEFAULT_PARAMS

headers =  {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + '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'
}

prompt = '''假设你是一个事件因果关系判断模型，事件关系类型有[前提，造成，没有关系]三种。我会给你一篇文章，并描述两个事件在文章中的位置以及触发词。 请你判断事件一和事件二之间的因果关系，并组成三元组，且形式为(event1, relation, event2)。

给定的文章为：["The Cherry Valley massacre was an attack by British and Iroquois forces on a fort and the village of Cherry Valley in eastern New York on November 11, 1778, during the American Revolutionary War.",
"It has been described as one of the most horrific frontier massacres of the war.",        
"A mixed force of Loyalists, British soldiers, Seneca and Mohawks descended on Cherry Valley, whose defenders, despite warnings, were unprepared for the attack.",
"During the raid, the Seneca in particular targeted non-combatants, and reports state that 30 such individuals were slain, in addition to a number of armed defenders.",
"The raiders were under the overall command of Walter Butler, who exercised little authority over the Indians on the expedition.",
"Historian Barbara Graymont describes Butler's command of the expedition as \"criminally incompetent\".",         
"The Seneca were angered by accusations that they had committed atrocities at the Battle of Wyoming, and the colonists' recent destruction of their forward bases of operation at Unadilla, Onaquaga, and Tioga.",
"Butler's authority with the Indians was undermined by his poor treatment of Joseph Brant, the leader of the Mohawks.",
"Butler repeatedly maintained, against accusations that he permitted the atrocities to take place, that he was powerless to restrain the Seneca.",         
"During the campaigns of 1778, Brant achieved an undeserved reputation for brutality.",         
"He was not present at Wyoming \u2014 although many thought he was \u2014 and he actively sought to minimize the atrocities that took place at Cherry Valley.",
"Diaries belonging to British soldiers during the campaign state the regiment as being the \"butchers\" and given that Butler was the overall commander of the expedition, there is controversy as to who actually ordered or failed to restrain the killings.",         
"The massacre contributed to calls for reprisals, leading to the 1779 Sullivan Expedition which drove the Iroquois out of western New York."
]

事件一：文章列表的第11个句子的took place触发了Process_start类型的事件

事件二：文章列表的第13个句子的drove触发了Motion类型的事件

在给定的文章中，事件一和事件二之间的因果关系是什么？请按照形式(event1, relation, event2)回答：'''


param['prompt'] = prompt


r = requests.post(url='https://api.openai.com/v1/completions',headers=headers, data = param)

print(r)
