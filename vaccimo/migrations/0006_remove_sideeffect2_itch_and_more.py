# Generated by Django 4.1.1 on 2023-05-07 13:48

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('vaccimo', '0005_user_datevaccinated_user_vaccination_brand2'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='sideeffect2',
            name='itch',
        ),
        migrations.RemoveField(
            model_name='sideeffect2',
            name='tenderness',
        ),
    ]