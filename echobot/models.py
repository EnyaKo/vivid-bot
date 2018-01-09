from django.db import models


class Personality(models.Model):
    p_type = models.TextField(null=False, primary_key=True)


class WeightedPersonality(models.Model):
    personality = models.ForeignKey(Personality)
    weight = models.FloatField(default=0)


class Phrase(models.Model):
    content = models.TextField(null=False, primary_key=True)
    weighted_personality_list = models.ManyToManyField(WeightedPersonality)

    class Meta:
        ordering = ('content',)


class BiGramRelation(models.Model):
    freq = models.IntegerField(default=0)
    prev = models.ForeignKey(Phrase, related_name="prev_phrase_in_bigram")
    post = models.ForeignKey(Phrase, related_name="post_phrase_in_bigram")


class TriGramRelation(models.Model):
    freq    = models.IntegerField(default=0)
    first   = models.ForeignKey(Phrase, related_name="first_phrase_in_trigram")
    mid     = models.ForeignKey(Phrase, related_name="mid_phrase_in_trigram")
    last    = models.ForeignKey(Phrase, related_name="last_phrase_in_trigram")
