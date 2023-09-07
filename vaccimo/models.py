from datetime import datetime
from distutils.command.upload import upload
from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
import datetime
import os
from django.utils import timezone

# Create your models here.
#User = settings.AUTH_USER_MODEL

def filepath(request,filename):
    old_filename = filename
    timeNow = datetime.datetime.now().strftime('%y%m%d%H:%M:%S')
    filename = "%s%s" % (timeNow,old_filename)
    return os.path.join('uploads/',filename)
#def filepath(instance, filename):
#    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
#    return 'user_{0}/{1}'.format(instance.id, filename)

class user(models.Model):
    id = models.AutoField(primary_key=True,)
    file = models.ImageField(upload_to=filepath, null=True, blank=True)
    email = models.CharField(max_length=100, default='N/A')
    name = models.CharField(max_length=100, default='N/A')
    suffix = models.CharField(max_length=100, default='N/A')
    nameFirst = models.CharField(max_length=100, default='N/A')
    nameLast = models.CharField(max_length=100, default='N/A')
    contact_number = models.CharField(max_length=100, default='N/A')
    firstDose = models.CharField(max_length=100, default='N/A')
    secondDose = models.CharField(max_length=100, default='N/A')
    firstBooster = models.CharField(max_length=100, default='N/A')
    secondBooster = models.CharField(max_length=100, default='N/A')
    address = models.CharField(max_length=100, default='N/A')
    age = models.CharField(max_length=100, default='N/A')
    dateVaccinated = models.CharField(max_length=100, default='N/A')
    dateVaccinated1 = models.CharField(max_length=100, default='N/A')
    dateVaccinated2 = models.CharField(max_length=100, default='N/A')
    dateVaccinated3 = models.CharField(max_length=100, default='N/A')
    vaccination_brand = models.CharField(max_length=100, default='N/A')
    vaccination_brand1 = models.CharField(max_length=100, default='N/A')
    vaccination_brand2 = models.CharField(max_length=100, default='N/A')
    vaccination_brand3 = models.CharField(max_length=100, default='N/A')    
    gender = models.CharField(max_length=100, default='N/A')
    status = models.CharField(max_length=10, blank=True, default='N/A')
    date_created = models.DateField(auto_now_add=True, null=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    def __str__(self):
        return self.name

    class Meta:
        db_table = "user"
        
class userRestorData(models.Model):
    id = models.AutoField(primary_key=True,)
    file = models.ImageField(upload_to=filepath, null=True, blank=True)
    email = models.CharField(max_length=100, null=True)
    name = models.CharField(max_length=100, null=True)
    suffix = models.CharField(max_length=100, default='')
    nameFirst = models.CharField(max_length=100, null=True)
    nameLast = models.CharField(max_length=100, null=True)
    contact_number = models.CharField(max_length=100, null=True)
    firstDose = models.CharField(max_length=100, null=True)
    secondDose = models.CharField(max_length=100, null=True)
    firstBooster = models.CharField(max_length=100, null=True)
    secondBooster = models.CharField(max_length=100, null=True)
    address = models.CharField(max_length=100, null=True)
    age = models.CharField(max_length=100, null=True)
    dateVaccinated = models.CharField(max_length=100, null=True)
    dateVaccinated1 = models.CharField(max_length=100, null=True)
    dateVaccinated2 = models.CharField(max_length=100, null=True)
    dateVaccinated3 = models.CharField(max_length=100, null=True)
    vaccination_brand = models.CharField(max_length=100, null=True)
    vaccination_brand1 = models.CharField(max_length=100, null=True)
    vaccination_brand2 = models.CharField(max_length=100, null=True)
    vaccination_brand3 = models.CharField(max_length=100, null=True)    
    gender = models.CharField(max_length=100, null=True)
    date_created = models.DateField(auto_now_add=True, null=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    def __str__(self):
        return self.name

    class Meta:
        db_table = "userRestorData"
                
class questioner (models.Model):

    id = models.AutoField(primary_key=True)
    Q0 = models.CharField(max_length=100, null=True)
    Q1 = models.CharField(max_length=100, null=True)
    Q2 = models.CharField(max_length=100, null=True)
    Q3 = models.CharField(max_length=100, null=True)
    Q4 = models.CharField(max_length=100, null=True)
    Q5 = models.CharField(max_length=100, null=True)
    Q6 = models.CharField(max_length=100, null=True)
    Q7 = models.CharField(max_length=100, null=True)
    Q8 = models.CharField(max_length=100, null=True)
    Q9 = models.CharField(max_length=100, null=True)
    Q10 = models.CharField(max_length=100, null=True)
    Q11 = models.CharField(max_length=100, null=True)
    Q12 = models.CharField(max_length=100, null=True)
    Q13 = models.CharField(max_length=100, null=True)
    Q14 = models.CharField(max_length=100, null=True)
    Q15 = models.CharField(max_length=100, null=True)
    Q16 = models.CharField(max_length=100, null=True)
    Q17 = models.CharField(max_length=100, null=True)
    Q18 = models.CharField(max_length=100, null=True)
    Q19 = models.CharField(max_length=100, null=True)
    Q20 = models.CharField(max_length=100, null=True)
    Q21 = models.CharField(max_length=100, null=True)
    Q22 = models.CharField(max_length=100, null=True)
    allergy2 = models.CharField(max_length=100, null=True)
    allergy3 = models.CharField(max_length=100, null=True)
    allergy4 = models.CharField(max_length=100, null=True)
    allergy5 = models.CharField(max_length=100, null=True)
    Q23 = models.CharField(max_length=100, null=True)
    Q24 = models.CharField(max_length=100, null=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    allergy = models.ForeignKey(user, on_delete=models.CASCADE)
    allergy1 = models.DateField(auto_now_add=True, null=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "questioner"


class sideeffect (models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, default='No')
    muscle_ache = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    redness = models.CharField(max_length=100, default='No')
    swelling = models.CharField(max_length=100, default='No')
    #warmth = models.CharField(max_length=100, default='No')
    chills = models.CharField(max_length=100, default='No')
    join_pain = models.CharField(max_length=100, default='No')
    fatigue = models.CharField(max_length=100, default='No')
    nausea = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    feverish = models.CharField(max_length=100, default='No')
    warmth = models.DateField(auto_now=True)
    induration = models.CharField(max_length=100, default='No')
    itch = models.ForeignKey(questioner, on_delete=models.CASCADE)
    tenderness = models.ForeignKey(user, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "sideeffect"

class firstdose (models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=100, default='0')
    InjectionSitePain = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    rashes = models.CharField(max_length=100, default='No')
    itchiness = models.CharField(max_length=100, default='No')
    cough = models.CharField(max_length=100, default='No')
    bodyPain = models.CharField(max_length=100, default='No')
    soarThroat = models.CharField(max_length=100, default='No')
    stomachAche = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    difficultyOfBreathing = models.CharField(max_length=100, default='No')
    chestPain = models.CharField(max_length=100, default='No')
    disorentation = models.CharField(max_length=100, default='No')
    tunnelVision = models.CharField(max_length=100, default='No')
    seizure = models.CharField(max_length=100, default='No')
    others = models.CharField(max_length=100, null=True)
    status = models.CharField(max_length=10, blank=True, null=True)
    date = models.DateField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    def __str__(self):
        return self.name

    class Meta:
        db_table = "firstdose"

class firstdoserestore (models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=100, default='0')
    InjectionSitePain = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    rashes = models.CharField(max_length=100, default='No')
    itchiness = models.CharField(max_length=100, default='No')
    cough = models.CharField(max_length=100, default='No')
    bodyPain = models.CharField(max_length=100, default='No')
    soarThroat = models.CharField(max_length=100, default='No')
    stomachAche = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    difficultyOfBreathing = models.CharField(max_length=100, default='No')
    chestPain = models.CharField(max_length=100, default='No')
    disorentation = models.CharField(max_length=100, default='No')
    tunnelVision = models.CharField(max_length=100, default='No')
    seizure = models.CharField(max_length=100, default='No')
    others = models.CharField(max_length=100, null=True)
    status = models.CharField(max_length=10, blank=True, null=True)
    date = models.DateField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    def __str__(self):
        return self.name

    class Meta:
        db_table = "firstdoserestore"
        
class seconddose (models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=100, default='0')
    InjectionSitePain = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    rashes = models.CharField(max_length=100, default='No')
    itchiness = models.CharField(max_length=100, default='No')
    cough = models.CharField(max_length=100, default='No')
    bodyPain = models.CharField(max_length=100, default='No')
    soarThroat = models.CharField(max_length=100, default='No')
    stomachAche = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    difficultyOfBreathing = models.CharField(max_length=100, default='No')
    chestPain = models.CharField(max_length=100, default='No')
    disorentation = models.CharField(max_length=100, default='No')
    tunnelVision = models.CharField(max_length=100, default='No')
    seizure = models.CharField(max_length=100, default='No')
    others = models.CharField(max_length=100, null=True)
    status = models.CharField(max_length=10, blank=True, null=True)
    date = models.DateField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "seconddose"

class seconddoserestore (models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=100, default='0')
    InjectionSitePain = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    rashes = models.CharField(max_length=100, default='No')
    itchiness = models.CharField(max_length=100, default='No')
    cough = models.CharField(max_length=100, default='No')
    bodyPain = models.CharField(max_length=100, default='No')
    soarThroat = models.CharField(max_length=100, default='No')
    stomachAche = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    difficultyOfBreathing = models.CharField(max_length=100, default='No')
    chestPain = models.CharField(max_length=100, default='No')
    disorentation = models.CharField(max_length=100, default='No')
    tunnelVision = models.CharField(max_length=100, default='No')
    seizure = models.CharField(max_length=100, default='No')
    others = models.CharField(max_length=100, null=True)
    status = models.CharField(max_length=10, blank=True, null=True)
    date = models.DateField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    def __str__(self):
        return self.name

    class Meta:
        db_table = "seconddoserestore"
        
class firstbooster (models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=100, default='0')
    InjectionSitePain = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    rashes = models.CharField(max_length=100, default='No')
    itchiness = models.CharField(max_length=100, default='No')
    cough = models.CharField(max_length=100, default='No')
    bodyPain = models.CharField(max_length=100, default='No')
    soarThroat = models.CharField(max_length=100, default='No')
    stomachAche = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    difficultyOfBreathing = models.CharField(max_length=100, default='No')
    chestPain = models.CharField(max_length=100, default='No')
    disorentation = models.CharField(max_length=100, default='No')
    tunnelVision = models.CharField(max_length=100, default='No')
    seizure = models.CharField(max_length=100, default='No')
    others = models.CharField(max_length=100, null=True)
    status = models.CharField(max_length=10, blank=True, null=True)
    date = models.DateField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "firstbooster"

class firstboosterrestore (models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=100, default='0')
    InjectionSitePain = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    rashes = models.CharField(max_length=100, default='No')
    itchiness = models.CharField(max_length=100, default='No')
    cough = models.CharField(max_length=100, default='No')
    bodyPain = models.CharField(max_length=100, default='No')
    soarThroat = models.CharField(max_length=100, default='No')
    stomachAche = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    difficultyOfBreathing = models.CharField(max_length=100, default='No')
    chestPain = models.CharField(max_length=100, default='No')
    disorentation = models.CharField(max_length=100, default='No')
    tunnelVision = models.CharField(max_length=100, default='No')
    seizure = models.CharField(max_length=100, default='No')
    others = models.CharField(max_length=100, null=True)
    status = models.CharField(max_length=10, blank=True, null=True)
    date = models.DateField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "firstboosterrestore"
        
class secondbooster (models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=100, default='0')
    InjectionSitePain = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    rashes = models.CharField(max_length=100, default='No')
    itchiness = models.CharField(max_length=100, default='No')
    cough = models.CharField(max_length=100, default='No')
    bodyPain = models.CharField(max_length=100, default='No')
    soarThroat = models.CharField(max_length=100, default='No')
    stomachAche = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    difficultyOfBreathing = models.CharField(max_length=100, default='No')
    chestPain = models.CharField(max_length=100, default='No')
    disorentation = models.CharField(max_length=100, default='No')
    tunnelVision = models.CharField(max_length=100, default='No')
    seizure = models.CharField(max_length=100, default='No')
    others = models.CharField(max_length=100, null=True)
    status = models.CharField(max_length=10, blank=True, null=True)
    date = models.DateField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "secondbooster"
class secondboosterrestored (models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=100, default='0')
    InjectionSitePain = models.CharField(max_length=100, default='No')
    headache = models.CharField(max_length=100, default='No')
    fever = models.CharField(max_length=100, default='No')
    rashes = models.CharField(max_length=100, default='No')
    itchiness = models.CharField(max_length=100, default='No')
    cough = models.CharField(max_length=100, default='No')
    bodyPain = models.CharField(max_length=100, default='No')
    soarThroat = models.CharField(max_length=100, default='No')
    stomachAche = models.CharField(max_length=100, default='No')
    vomiting = models.CharField(max_length=100, default='No')
    difficultyOfBreathing = models.CharField(max_length=100, default='No')
    chestPain = models.CharField(max_length=100, default='No')
    disorentation = models.CharField(max_length=100, default='No')
    tunnelVision = models.CharField(max_length=100, default='No')
    seizure = models.CharField(max_length=100, default='No')
    others = models.CharField(max_length=100, null=True)
    status = models.CharField(max_length=10, blank=True, null=True)
    date = models.DateField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "secondboosterrestore"        
# class User(AbstractUser):
#     is_admin = models.BooleanField(default=False)
#     is_customer = models.BooleanField(default=False)
