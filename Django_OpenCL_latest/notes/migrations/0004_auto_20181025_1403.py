# -*- coding: utf-8 -*-
# Generated by Django 1.11.15 on 2018-10-25 14:03
from __future__ import unicode_literals

from django.db import migrations, models
import notes.models


class Migration(migrations.Migration):

    dependencies = [
        ('notes', '0003_post_selected_algo'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='post',
            name='h_type',
        ),
        migrations.AddField(
            model_name='post',
            name='inputfile_failure',
            field=models.FileField(blank=True, null=True, upload_to=notes.models.user_directory_path),
        ),
        migrations.AddField(
            model_name='post',
            name='no_days_failure',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
