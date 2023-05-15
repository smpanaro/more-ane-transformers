# Quantitative Comparison

The conversion from the original model weights to CoreML is lossy. One way to measure the loss is using Peak Signal to Noise Ratio as in [ml-ane-transformers](https://github.com/apple/ml-ane-transformers/blob/da64000fa56cc85b0859bc17cb16a3d753b8304a/ane_transformers/huggingface/test_distilbert.py#L173). Values above 60 are considered faithful reconstructions of the original model.

<details>
<summary>gpt2</summary>

|gpt2 (124M)|gpt2-medium (350M)|gpt2-large (774M)|gpt2-xl (1558M)|
|-          |-                 |-                |-              |
|60.2       |58.2              |68.1             |67.4           |

</details>

<details>
<summary>pythia</summary>

|70M |160M|410M|1B  |1.4B|2.8B|6.9B|
|-   |-   |-   |-   |-   |-   |-   |
|56.5|59.7|34.2|63.6|45  |43.4|41.2|

<sub>values &lt;&lt;60 all have more than 16 layers, still seem usable (see below)</sub>
<sub>70M+160M are unstable on CPU only, CPU+NE is required</sub>

</details>

# Text Generation Examples

Non-cherry picked examples using the original gpt2 "out of distribution" prompt for 350 tokens with temperature 0.85 and top k 80.

Prompt:
```
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.
```

## gpt2

<details>
<summary>gpt2 124M</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The researchers, from a team led by Dr. Zhen Wang from the Wuhan Institute for Animal Sciences, published their results today (Aug. 17) in the same journal.

"Our results, showing that there are many unicorns in North America today, are a step towards bringing unicorns to the American landscape," said Wang, who is also head of the Chinese University of Natural Science in Beijing and a co-author on the study.

"It offers hope that the unicorns will be able to spread to other parts of North America before long," he added.

The research was part of an ongoing project by the GCSF-funded international unicorns project, created as part of the National Center for the Study of Natural Resource, which aims to use "new information about animals and plants to discover new ways to produce and consume natural resources."

Previous studies have shown that people are aware of the species and how it communicates. But the researchers found that the unicorns also used social media to communicate more effectively than other animals.

While unicorns are less common in North America and other parts of Asia, their habitat has been exploited by humans for agriculture, for medicinal purposes, as well as for fishing and livestock feed.

"They have been caught in numerous places and have been so hard to find," Wang said.

"Because they are more widespread and the species is so rare, we were almost stunned when we found them on our farm, and so we were surprised to learn that they also communicate," he added. "There is a mystery here about how these animals communicate, but we're very excited. It's very exciting."

Explore further: Researchers discover
```
</details>

<details>
<summary>gpt2-medium 350M</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. "They don't even know their own language," said Dr. Gomes as he explains to Telegraph.com how the little unicorns, called Anasazi, are able to communicate with each other.

"They are very intelligent people, they know how to look for the source," said Dr. Gomes. "The species is in trouble, but they are doing the best they can with what they have."

In 2009, Dr. Gomes and his colleagues discovered their unusual creatures in the region named Zaregaz. Six years later, they released two of them into the wild. The pair were introduced into the wild in 2002 — at a time when several other species had been discovered nearby.

Dr. Gomes and his team are currently trying to get the others to breed with the herd so that they can fulfill the idealistic dream of having a place where they can live a peaceful existence, while also protecting other areas, from the elements to hunting animals. "They live off their natural environment, from the plants. They live with animals without killing," he said. He added that the habitat is well protected. The animals themselves live in groups, the pair say, making them extremely docile and docile animals. Nevertheless, the researchers said the animals could easily be used for the breeding, and the process could possibly begin soon.

This is not the first time that Dr. Gomes has sought to prove the superiority of humans over these tiny, native creatures. In 2009, he and his colleagues discovered the existence of a herd of humans, or Asymptoceratops, in Brazil's Barribas, and they named them Zorro.

Do you think that some
```
</details>

<details>
<summary>gpt2-large 774M</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

A team of geologists and animal experts discovered the unicorn herd in a remote valley in Colombia. They decided to keep it a secret from the public, as it hasn't been discovered before.

Teddy Katz, head of the Canadian School of Tropical Biology and National Geographic Explorer-in-Residence, told The Huffington Post Canada that even though the unicorns weren't aggressive, they were in the dark.

SPONSORED

"There was no one looking after them, to say the least," Katz said. "We did find out it was a pretty small herd, so I'm sure they didn't have a lot of space to walk."

The scientists were surprised to discover the herd of tiny unicorns. "They're just tiny, and they're very, very shy," Katz said. "It took me about 10 seconds to recognize them."

The researchers found a huge herd of zebras and kangaroos as well as several other animals. They also found the unicorns with their head shaved, which suggests that they might have an interesting relationship with their habitat, Katz said.

"They've got a special relationship with the Andean glaciers," he said. "They're a source of water, they make up for what they lose in temperatures in winter, and as a result they've been able to keep going for so long."

During the winter, snow falls and when the glaciers melt, they deposit a layer of ice over the surface of the mountains. This year, the glaciers broke through the ice and the new layer of snow hit the surface, providing an abundance of water for the unicorns, making them even more comfortable.

"We found that during
```
</details>

<details>
<summary>gpt2-xl 1558M</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The scientists, from the University of California, Los Angeles, spent two years studying the environment of the valley, and have discovered the unicorns' "language."

The researchers say the unicorns spoke in perfect English, which, according to them, is not only the world's first, but also the only language spoken by unicorns.

"These amazing unicorns are known as plectrae, a species of astragalus, and they are the only known example in the animal kingdom that knows and uses a grammar," said University of California at Santa Barbara physicist, Dr. David Klee. "They are so unique and bizarre that they defy all description."

The team believes it is almost impossible to speak Latin or any of the other thousands of languages spoken in Latin America, including Portuguese, Spanish, French, and Italian, and to have a human-like ability to communicate.

According to the researchers, the unicorns speak using a syntax all their own. When the scientists asked the unicorns questions, it was clear that they were speaking fluent English.

The scientists believe the unicorns are from the plectrae but can speak English because of their physiology. The unicorns have an extra layer of tissue around their vocal cords that allows them to store sounds. When the scientists played the tones of English for the animals, the sounds were heard.

"Only a few human languages can be understood without having a good first language," said Dr. Klee.

The unicorns also knew a great deal about their environment. The local flora was not just in the valley but throughout the entire mountain range.

"It was kind of like a big science experiment
```
</details>

## pythia

<details>
<summary>pythia-70m</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

Scientists say that their study is not just an unicorns’ research technique but a way to explain how the plant can penetrate deeper into the skin. Scientists point out that a key ingredient in many new tools is the use of a chemical that can help create a layer of a body. Scientists are working on how to make something that is actually a body.

While the study appears to have a “strong appeal” in the region, people who saw the fruit in the bag, and most who knew nothing about it, don’t really understand the natural nature of the plant. Scientists think that’s not an easy task. This is where scientists call themselves scientists.

These scientists are very proud of the study, but I’m not a scientist. As the research team develops, the researchers become more specific that their work is a matter of how we apply and how we do it. The research team then attempts to provide an argument for a particular experiment, and while they are in denial of the safety of their findings, the scientists are more likely to be skeptical about the findings themselves.

Scientists at the University of Chicago’s UC Berkeley have been using these methods for years to find out whether the plant can actually penetrate deeper into the skin of the animals. Of course, scientists want to know how that can be done and how you can look at it.

On a recent visit, the scientist says that the researchers’ findings are actually “just a tool to help scientists to understand how the plant can penetrate deeper into the skin of animals.” And it’s not just about how the plant can penetrate deeper into the skin of the cow. It’s about how we create our own skin.
```
</details>

<details>
<summary>pythia-160m</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

"Nay, I'm not saying this is the first time I've ever heard of it," I thought.

"I must be wrong. For the past three days, I've been taking a dip in the Himalayas and the sun was glowing in the sky. When I arrived, a few days later, I stood up and walked out into the valley. As I passed, the fire filled with the salt water of the lake. As the water had gone completely black, the wind shook it open and I saw the ghost of a bird. I was too exhausted to make much sound but there was a feeling that I was here, I think.

"The sun was high and it wasn't all that bright, but the clouds were bright and it was a very pleasant day to be home. We'd been riding the stream in the mountain village, with the wind so thick, and there were a few clouds in the sky. I thought the first time I saw them, I was terrified. The fire was full and I could feel the vibrations of the air around me. I went for a walk, and it was a wonderful day. There were some people in the house. I knew the women from the village and they were the only ones in the group who'd heard of the land. I also knew that the village's name was in fact 'Oye' and they all belonged to the same village as my uncle and father and that they all came from another side of the mountain. They were the same people I knew in the village and I was so lonely.

"I took some milk and dried some fruits, and I knew that the moon was in the sky, so I began to
```
</details>

<details>
<summary>pythia-410m</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. A scientist discovered a herd of unicorns living in a remote, previously unexplored valley in the Andes Mountains, in the United States. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

Although some unicorns can be extremely playful, the research group was even more surprised to find out that the unicorns were actually quite dull, to the point that they were actually considered to be petrified.

Their study was published today in the journal Proceedings of the Royal Society B.

What makes a unicorn so interesting is that not one person in this article has ever heard of them.

“This unicorn breed seems very elusive, and has only been documented for about a half-dozen years,” said lead author J.S. Linder. “Our research team is able to find a number of unicorns in the Andes Mountains in Peru and the United States, but only one in Bolivia. This is an extremely important discovery because it means that these unicorns are not only very well-known, but they are also highly valued.”

“Their ability to communicate in perfect English makes them an incredibly interesting pet,” noted co-author Andrew J. Egan of the University of California, Irvine. “This unicorn is also highly social, living in a very solitary environment where it is a rare sight. When it hears its name, it quickly becomes very friendly, and is able to communicate with other unicorns.”

Linder said she hopes to learn more about this unicorn very soon, so she can help document it and give it a home in the future.<|endoftext|>Q:

Does SQL Server support
```
</details>

<details>
<summary>pythia-1b</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The scientific team of professors Dr. Aydin Ehsan and Dr. Rehana Ghassemi revealed the fact that they were able to translate the unicorns' English into the most commonly spoken language on the planet.

The results also revealed that the unicorns' English ability was comparable with that of the native English speakers.

It also revealed that the unicorns' English proficiency was actually greater than all other animals.

Researchers even translated their own English with the unicorns' language from the Andes Mountains.

It also confirmed that these unicorns had a remarkable ability in communicating with their fellow animals.

The unicorns have lived in the jungle for hundreds of years and the scientists have learned a great amount from them.

The team even learned that the unicorns were able to understand the concept of love and emotions.

The researchers also revealed that the unicorns had an ability to sense the difference between human beings and other animals.

The scientists even learned that the unicorns' understanding of the "animal" language is far more advanced than that of other animals.

"We took advantage of the unicorns' rare ability to speak English to find out whether there is any way of identifying who or what a unicorn is," the researches said.

The study also revealed that the unicorns are very creative in communicating with their fellow animals.

The scientists noted that the unicorns' communication skills were very impressive and they even learned the unicorn language through their own imagination.<|endoftext|>The only thing that has stopped me from playing is the fact that the system is broken and
```
</details>

<details>
<summary>pythia-1.4b</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. A video of the event shows the horses doing a ‘cow-o’-the-hoof pose, accompanied by a French-English flute-play that we hope we can hear at some point soon.

“What we didn’t expect was for the unicorns to speak English,” said lead investigator, Marcel Séguin, of the University of Lyon’s Department of Agronomy. “We were so surprised we thought they would have no language abilities at all, as they are known to be very intelligent.”

Séguin, along with a team of researchers from Lyon, France, worked in the Percho Valley in Ecuador, and collected data from the horses’ hooves, hoof and hoof-bone. The horses were part of a three-year study funded by the European Research Council and the French Ministry of Agriculture, which wanted to find out if unicorns could have language abilities. While some of the other horses had the ability to speak, the horses in the study had their own dialect similar to that of humans, the researchers said.

“They are able to learn and understand a variety of human languages, but they cannot speak or hear it,” Séguin said. “Humans can learn a few words of a foreign language, but it’s not enough to communicate with other humans.”

The researchers spent months exploring the valley and filming the horses, which they discovered were only one in a herd of 1,000. They said they were shocked to find so many horses in just one isolated zone. “It’s surprising that we found such a large number of horses in such a remote area,” Séguin said.
```
</details>

<details>
<summary>pythia-2.8b (generated on GPU)</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. This is almost certainly the first time that such a discovery has been made in human history, and the first time that the unicorns' language is known. Unicorns are said to be 'fawns' that are able to speak the language of the human race with ease.

"Unfortunately, I was unable to speak with them to find out why they were living there and where exactly they came from. But I will tell you one thing: it is safe to say that they were not accidentally found living there. This is the only place where unicorns live among humans. It was clearly a deliberate act." Dr Toni Emanuele, the scientist who has made this discovery.

This report is believed to be the most important scientific discovery in the history of mankind. Unicorns are said to have been seen in the past, but never before has a single unicorn spoken to another human. "It certainly is a very interesting discovery," says the head of the United Nations General Assembly, Dr Herman Coats. "It is very, very rare for a unicorn to be able to speak to humans." It is believed that this discovery will be considered in the United Nations next year, and that discussions will take place in order to consider changing the rules on the existence of unicorns in order to make sure that they can live peacefully with humans.

"I have waited for this day for almost 20 years," says Dr Emanuele. "I have been waiting for years in order to discover the secret of unicorns, the secret of their life. I have never been so excited as I was listening to this radio interview."<|endoftext|>"Previously on "The Vampire Di
```
</details>

<details>
<summary>pythia-6.9b (generated on GPU)</summary>

```
[Prompt] In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. The new breed of unicorns found in the Amazon jungle appeared to have originated with the domesticated European horse, reports the Telegraph.

The unicorns are called Chama, after the Chama River basin where they live, and according to a report in the Daily Mail, they have been known to the indigenous people of the Amazon, called the "Shuar". They were only discovered in 2007 by a team of scientists from Liverpool John Moores University, led by Professor Stephen LeComber.

Researchers have named the species of unicorn Chama, after its native river basin. “They are incredibly rare, endangered animals, and they live in a very remote part of the Andes,” Professor Stephen LeComber said. According to the Daily Mail, the unicorns are known as the “Chama”, after the Chama River basin where they live.

They are known by the Shuar people of the Amazon, and live in a remote, previously unexplored valley in the Andes. These “Chama” unicorns, however, are not the same as the legendary, mythical unicorns of Europe, just as their namesake river basin are not the same as in the rest of the world.

The European unicorn myth was believed to have originated in Celtic folklore. As a result, an unicorn was believed to represent a being that was half human and half “spirit animal” and the myth of the unicorn was used as a way of promoting the union of a human and a “spirit animal”. Today, the unicorn has become a symbol of romantic love.

“As a species, unicorn fossils have been found in Egypt,
```
</details>