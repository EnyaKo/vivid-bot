# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2016-11-21 14:14
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Personality',
            fields=[
                ('p_type', models.TextField(primary_key=True, serialize=False)),
            ],
        ),
        migrations.CreateModel(
            name='Phrase',
            fields=[
                ('content', models.TextField(primary_key=True, serialize=False)),
                ('personality', models.ManyToManyField(to='echobot.Personality')),
            ],
            options={
                'ordering': ('content',),
            },
        ),
        migrations.CreateModel(
            name='PhraseRelation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('freq', models.IntegerField(default=0)),
                ('post', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='post_phrase_in_relation', to='echobot.Phrase')),
                ('prev', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='prev_phrase_in_relation', to='echobot.Phrase')),
            ],
        ),
    ]
