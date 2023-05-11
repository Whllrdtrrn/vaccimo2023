# Generated by Django 4.1.1 on 2023-05-11 09:29

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import vaccimo.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='questioner',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('Q0', models.CharField(max_length=100, null=True)),
                ('Q1', models.CharField(max_length=100, null=True)),
                ('Q2', models.CharField(max_length=100, null=True)),
                ('Q3', models.CharField(max_length=100, null=True)),
                ('Q4', models.CharField(max_length=100, null=True)),
                ('Q5', models.CharField(max_length=100, null=True)),
                ('Q6', models.CharField(max_length=100, null=True)),
                ('Q7', models.CharField(max_length=100, null=True)),
                ('Q8', models.CharField(max_length=100, null=True)),
                ('Q9', models.CharField(max_length=100, null=True)),
                ('Q10', models.CharField(max_length=100, null=True)),
                ('Q11', models.CharField(max_length=100, null=True)),
                ('Q12', models.CharField(max_length=100, null=True)),
                ('Q13', models.CharField(max_length=100, null=True)),
                ('Q14', models.CharField(max_length=100, null=True)),
                ('Q15', models.CharField(max_length=100, null=True)),
                ('Q16', models.CharField(max_length=100, null=True)),
                ('Q17', models.CharField(max_length=100, null=True)),
                ('Q18', models.CharField(max_length=100, null=True)),
                ('Q19', models.CharField(max_length=100, null=True)),
                ('Q20', models.CharField(max_length=100, null=True)),
                ('Q21', models.CharField(max_length=100, null=True)),
                ('Q22', models.CharField(max_length=100, null=True)),
                ('allergy2', models.CharField(max_length=100, null=True)),
                ('allergy3', models.CharField(max_length=100, null=True)),
                ('allergy4', models.CharField(max_length=100, null=True)),
                ('allergy5', models.CharField(max_length=100, null=True)),
                ('Q23', models.CharField(max_length=100, null=True)),
                ('Q24', models.CharField(max_length=100, null=True)),
                ('allergy1', models.DateField(auto_now_add=True, null=True)),
            ],
            options={
                'db_table': 'questioner',
            },
        ),
        migrations.CreateModel(
            name='userRestorData',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('file', models.ImageField(blank=True, null=True, upload_to=vaccimo.models.filepath)),
                ('email', models.CharField(max_length=100, null=True)),
                ('name', models.CharField(max_length=100, null=True)),
                ('suffix', models.CharField(default='N/A', max_length=100)),
                ('nameFirst', models.CharField(max_length=100, null=True)),
                ('nameLast', models.CharField(max_length=100, null=True)),
                ('contact_number', models.CharField(max_length=100, null=True)),
                ('firstDose', models.CharField(max_length=100, null=True)),
                ('secondDose', models.CharField(max_length=100, null=True)),
                ('firstBooster', models.CharField(max_length=100, null=True)),
                ('secondBooster', models.CharField(max_length=100, null=True)),
                ('address', models.CharField(max_length=100, null=True)),
                ('age', models.CharField(max_length=100, null=True)),
                ('dateVaccinated', models.CharField(max_length=100, null=True)),
                ('dateVaccinated1', models.CharField(max_length=100, null=True)),
                ('dateVaccinated2', models.CharField(max_length=100, null=True)),
                ('dateVaccinated3', models.CharField(max_length=100, null=True)),
                ('vaccination_brand', models.CharField(max_length=100, null=True)),
                ('vaccination_brand1', models.CharField(max_length=100, null=True)),
                ('vaccination_brand2', models.CharField(max_length=100, null=True)),
                ('vaccination_brand3', models.CharField(max_length=100, null=True)),
                ('gender', models.CharField(max_length=100, null=True)),
                ('date_created', models.DateField(auto_now_add=True, null=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'userRestorData',
            },
        ),
        migrations.CreateModel(
            name='user',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('file', models.ImageField(blank=True, null=True, upload_to=vaccimo.models.filepath)),
                ('email', models.CharField(max_length=100, null=True)),
                ('name', models.CharField(max_length=100, null=True)),
                ('suffix', models.CharField(default='N/A', max_length=100)),
                ('nameFirst', models.CharField(max_length=100, null=True)),
                ('nameLast', models.CharField(max_length=100, null=True)),
                ('contact_number', models.CharField(max_length=100, null=True)),
                ('firstDose', models.CharField(max_length=100, null=True)),
                ('secondDose', models.CharField(max_length=100, null=True)),
                ('firstBooster', models.CharField(max_length=100, null=True)),
                ('secondBooster', models.CharField(max_length=100, null=True)),
                ('address', models.CharField(max_length=100, null=True)),
                ('age', models.CharField(max_length=100, null=True)),
                ('dateVaccinated', models.CharField(max_length=100, null=True)),
                ('dateVaccinated1', models.CharField(max_length=100, null=True)),
                ('dateVaccinated2', models.CharField(max_length=100, null=True)),
                ('dateVaccinated3', models.CharField(max_length=100, null=True)),
                ('vaccination_brand', models.CharField(max_length=100, null=True)),
                ('vaccination_brand1', models.CharField(max_length=100, null=True)),
                ('vaccination_brand2', models.CharField(max_length=100, null=True)),
                ('vaccination_brand3', models.CharField(max_length=100, null=True)),
                ('gender', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date_created', models.DateField(auto_now_add=True, null=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'user',
            },
        ),
        migrations.CreateModel(
            name='sideeffect',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(default='No', max_length=100)),
                ('muscle_ache', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('redness', models.CharField(default='No', max_length=100)),
                ('swelling', models.CharField(default='No', max_length=100)),
                ('chills', models.CharField(default='No', max_length=100)),
                ('join_pain', models.CharField(default='No', max_length=100)),
                ('fatigue', models.CharField(default='No', max_length=100)),
                ('nausea', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('feverish', models.CharField(default='No', max_length=100)),
                ('warmth', models.DateField(auto_now=True)),
                ('induration', models.CharField(default='No', max_length=100)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('itch', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vaccimo.questioner')),
                ('tenderness', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vaccimo.user')),
            ],
            options={
                'db_table': 'sideeffect',
            },
        ),
        migrations.CreateModel(
            name='seconddoserestore',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user', models.CharField(default='0', max_length=100)),
                ('InjectionSitePain', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('rashes', models.CharField(default='No', max_length=100)),
                ('itchiness', models.CharField(default='No', max_length=100)),
                ('cough', models.CharField(default='No', max_length=100)),
                ('bodyPain', models.CharField(default='No', max_length=100)),
                ('soarThroat', models.CharField(default='No', max_length=100)),
                ('stomachAche', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('difficultyOfBreathing', models.CharField(default='No', max_length=100)),
                ('chestPain', models.CharField(default='No', max_length=100)),
                ('disorentation', models.CharField(default='No', max_length=100)),
                ('tunnelVision', models.CharField(default='No', max_length=100)),
                ('seizure', models.CharField(default='No', max_length=100)),
                ('others', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date', models.DateField(auto_now=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'seconddoserestore',
            },
        ),
        migrations.CreateModel(
            name='seconddose',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user', models.CharField(default='0', max_length=100)),
                ('InjectionSitePain', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('rashes', models.CharField(default='No', max_length=100)),
                ('itchiness', models.CharField(default='No', max_length=100)),
                ('cough', models.CharField(default='No', max_length=100)),
                ('bodyPain', models.CharField(default='No', max_length=100)),
                ('soarThroat', models.CharField(default='No', max_length=100)),
                ('stomachAche', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('difficultyOfBreathing', models.CharField(default='No', max_length=100)),
                ('chestPain', models.CharField(default='No', max_length=100)),
                ('disorentation', models.CharField(default='No', max_length=100)),
                ('tunnelVision', models.CharField(default='No', max_length=100)),
                ('seizure', models.CharField(default='No', max_length=100)),
                ('others', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date', models.DateField(auto_now=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'seconddose',
            },
        ),
        migrations.CreateModel(
            name='secondboosterrestored',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user', models.CharField(default='0', max_length=100)),
                ('InjectionSitePain', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('rashes', models.CharField(default='No', max_length=100)),
                ('itchiness', models.CharField(default='No', max_length=100)),
                ('cough', models.CharField(default='No', max_length=100)),
                ('bodyPain', models.CharField(default='No', max_length=100)),
                ('soarThroat', models.CharField(default='No', max_length=100)),
                ('stomachAche', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('difficultyOfBreathing', models.CharField(default='No', max_length=100)),
                ('chestPain', models.CharField(default='No', max_length=100)),
                ('disorentation', models.CharField(default='No', max_length=100)),
                ('tunnelVision', models.CharField(default='No', max_length=100)),
                ('seizure', models.CharField(default='No', max_length=100)),
                ('others', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date', models.DateField(auto_now=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'secondboosterrestore',
            },
        ),
        migrations.CreateModel(
            name='secondbooster',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user', models.CharField(default='0', max_length=100)),
                ('InjectionSitePain', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('rashes', models.CharField(default='No', max_length=100)),
                ('itchiness', models.CharField(default='No', max_length=100)),
                ('cough', models.CharField(default='No', max_length=100)),
                ('bodyPain', models.CharField(default='No', max_length=100)),
                ('soarThroat', models.CharField(default='No', max_length=100)),
                ('stomachAche', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('difficultyOfBreathing', models.CharField(default='No', max_length=100)),
                ('chestPain', models.CharField(default='No', max_length=100)),
                ('disorentation', models.CharField(default='No', max_length=100)),
                ('tunnelVision', models.CharField(default='No', max_length=100)),
                ('seizure', models.CharField(default='No', max_length=100)),
                ('others', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date', models.DateField(auto_now=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'secondbooster',
            },
        ),
        migrations.AddField(
            model_name='questioner',
            name='allergy',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vaccimo.user'),
        ),
        migrations.AddField(
            model_name='questioner',
            name='author',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.CreateModel(
            name='firstdoserestore',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user', models.CharField(default='0', max_length=100)),
                ('InjectionSitePain', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('rashes', models.CharField(default='No', max_length=100)),
                ('itchiness', models.CharField(default='No', max_length=100)),
                ('cough', models.CharField(default='No', max_length=100)),
                ('bodyPain', models.CharField(default='No', max_length=100)),
                ('soarThroat', models.CharField(default='No', max_length=100)),
                ('stomachAche', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('difficultyOfBreathing', models.CharField(default='No', max_length=100)),
                ('chestPain', models.CharField(default='No', max_length=100)),
                ('disorentation', models.CharField(default='No', max_length=100)),
                ('tunnelVision', models.CharField(default='No', max_length=100)),
                ('seizure', models.CharField(default='No', max_length=100)),
                ('others', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date', models.DateField(auto_now=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'firstdoserestore',
            },
        ),
        migrations.CreateModel(
            name='firstdose',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('users', models.CharField(default='0', max_length=100)),
                ('InjectionSitePain', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('rashes', models.CharField(default='No', max_length=100)),
                ('itchiness', models.CharField(default='No', max_length=100)),
                ('cough', models.CharField(default='No', max_length=100)),
                ('bodyPain', models.CharField(default='No', max_length=100)),
                ('soarThroat', models.CharField(default='No', max_length=100)),
                ('stomachAche', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('difficultyOfBreathing', models.CharField(default='No', max_length=100)),
                ('chestPain', models.CharField(default='No', max_length=100)),
                ('disorentation', models.CharField(default='No', max_length=100)),
                ('tunnelVision', models.CharField(default='No', max_length=100)),
                ('seizure', models.CharField(default='No', max_length=100)),
                ('others', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date', models.DateField(auto_now=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('userA', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='vaccimo.user')),
            ],
            options={
                'db_table': 'firstdose',
            },
        ),
        migrations.CreateModel(
            name='firstboosterrestore',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user', models.CharField(default='0', max_length=100)),
                ('InjectionSitePain', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('rashes', models.CharField(default='No', max_length=100)),
                ('itchiness', models.CharField(default='No', max_length=100)),
                ('cough', models.CharField(default='No', max_length=100)),
                ('bodyPain', models.CharField(default='No', max_length=100)),
                ('soarThroat', models.CharField(default='No', max_length=100)),
                ('stomachAche', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('difficultyOfBreathing', models.CharField(default='No', max_length=100)),
                ('chestPain', models.CharField(default='No', max_length=100)),
                ('disorentation', models.CharField(default='No', max_length=100)),
                ('tunnelVision', models.CharField(default='No', max_length=100)),
                ('seizure', models.CharField(default='No', max_length=100)),
                ('others', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date', models.DateField(auto_now=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'firstboosterrestore',
            },
        ),
        migrations.CreateModel(
            name='firstbooster',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user', models.CharField(default='0', max_length=100)),
                ('InjectionSitePain', models.CharField(default='No', max_length=100)),
                ('headache', models.CharField(default='No', max_length=100)),
                ('fever', models.CharField(default='No', max_length=100)),
                ('rashes', models.CharField(default='No', max_length=100)),
                ('itchiness', models.CharField(default='No', max_length=100)),
                ('cough', models.CharField(default='No', max_length=100)),
                ('bodyPain', models.CharField(default='No', max_length=100)),
                ('soarThroat', models.CharField(default='No', max_length=100)),
                ('stomachAche', models.CharField(default='No', max_length=100)),
                ('vomiting', models.CharField(default='No', max_length=100)),
                ('difficultyOfBreathing', models.CharField(default='No', max_length=100)),
                ('chestPain', models.CharField(default='No', max_length=100)),
                ('disorentation', models.CharField(default='No', max_length=100)),
                ('tunnelVision', models.CharField(default='No', max_length=100)),
                ('seizure', models.CharField(default='No', max_length=100)),
                ('others', models.CharField(max_length=100, null=True)),
                ('status', models.CharField(blank=True, max_length=10, null=True)),
                ('date', models.DateField(auto_now=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'firstbooster',
            },
        ),
    ]
