from torch.utils.data import DataLoader
import torch
from email import message
from multiprocessing import context
from urllib import response
from venv import create
import time
from django.utils import timezone

from django.shortcuts import render, redirect
from django.template import loader
from django.contrib import auth
from django.contrib.auth import logout, authenticate
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from .models import user
from django.urls import reverse
import os
from django.http import FileResponse
from .filters import OrderFilter, EffectFilter
from .models import sideeffect
from .models import questioner
from .models import userRestorData, secondboosterrestored, firstboosterrestore, seconddoserestore, firstdoserestore
from .models import firstdose
from .models import seconddose
from .models import firstbooster
from .models import secondbooster
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.conf.urls.static import static
# import tensorflow as tf
import pandas as pd
from pandas.api.types import is_object_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import base64
from io import BytesIO

from django.utils.timezone import now
from django.db.models import Q
from django.db.models import Count
from django.db.models.functions import TruncDate
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from django.forms.models import model_to_dict

# pdf
from django.template.loader import get_template
from django.http import HttpResponse
from xhtml2pdf import pisa

# email verification
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from .tokens import account_activation_token
from django.core.mail import EmailMessage
from django.shortcuts import get_object_or_404
import warnings
warnings.filterwarnings('ignore')

# from google.colab import drive # for google drive access
# drive.mount('/content/drive')

# Create your views here.
# def preprocessingOOOO(request):

#     if bool(request.FILES.get('document', False)) == True:
#         uploaded_file = request.FILES['document']
#         name = uploaded_file.name
#         request.session['name'] = name
#         df = pd.read_csv(uploaded_file)
#         dataFrame = df.to_json()
#         request.session['df'] = dataFrame

#         rows = len(df.index)
#         request.session['rows'] = rows
#         header = df.axes[1].values.tolist()
#         request.session['header'] = header

#         attributes = len(header)
#         types = []
#         maxs = []
#         mins = []
#         means = []
#         # statistic attribut
#         for i in range(len(header)):
#             types.append(df[header[i]].dtypes)
#             if df[header[i]].dtypes != 'object':
#                 maxs.append(df[header[i]].max())
#                 mins.append(df[header[i]].min())
#                 means.append(round(df[header[i]].mean(),2))
#             else:
#                 maxs.append(0)
#                 mins.append(0)
#                 means.append(0)

#         zipped_data = zip(header, types, maxs, mins, means)
#         print(maxs)
#         datas = df.values.tolist()
#         data ={
#                 "header": header,
#                 "headers": json.dumps(header),
#                 "name": name,
#                 "attributes": attributes,
#                 "rows": rows,
#                 "zipped_data": zipped_data,
#                 'df': datas,
#                 "type": types,
#                 "maxs": maxs,
#                 "mins": mins,
#                 "means": means,
#             }
#     else:
#         name = 'None'
#         attributes = 'None'
#         rows = 'None'
#         data ={
#                 "name": name,
#                 "attributes": attributes,
#                 "rows": rows,
#             }
#     return render(request, 'system/index.html', data)
    
def generate_pdf(request, pk):
    # Get the object with the corresponding ID
    obj = user.objects.get(pk=pk)
    # Render the HTML template with the object data
    template = get_template('home/pdfReport.html')
    html = template.render({'obj': obj})
    # Generate the PDF file
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'
    pisa_status = pisa.CreatePDF(html, dest=response, pagesize='legal')
    
    if pisa_status.err:
        return HttpResponse('An error occurred while creating the PDF file')
    return response

def pdfAllReport(request):
    # Get the object with the corresponding ID
    obj = sideeffect.objects.filter(name=1)
    # Render the HTML template with the object data
    template = get_template('home/pdfAllReport.html')
    html = template.render({'obj': obj})
    # Generate the PDF file
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="AllReport.pdf"'
    pisa_status = pisa.CreatePDF(html, dest=response)
    if pisa_status.err:
        return HttpResponse('An error occurred while creating the PDF file')
    return response

def LayoutHome(request):
    return render(request, 'Layout/LayoutHome.html')


def LayoutIndex(request):
    prod = user()
    side = sideeffect()
    ques = questioner()
    if request.method == 'POST':
        prod.name = request.POST.get('name')
        prod.contact_number = request.POST.get('contact_number')
        prod.vaccination_brand = request.POST.get('vaccination_brand')
        prod.vaccination_site = request.POST.get('vaccination_site')
        prod.address = request.POST.get('address')
        prod.age = request.POST.get('age')
        prod.bday = request.POST.get('bday')
        prod.gender = request.POST.get('gender')
        side.muscle_ache = request.POST.get('muscle_ache')
        side.headache = request.POST.get('headache')
        side.fever = request.POST.get('fever')
        side.redness = request.POST.get('redness')
        side.swelling = request.POST.get('swelling')
        side.tenderness = request.POST.get('tenderness')
        side.warmth = request.POST.get('warmth')
        side.itch = request.POST.get('itch')
        side.induration = request.POST.get('induration')
        side.feverish = request.POST.get('feverish')
        side.chills = request.POST.get('chills')
        side.join_pain = request.POST.get('join_pain')
        side.fatigue = request.POST.get('fatigue')
        side.nausea = request.POST.get('nausea')
        side.vomiting = request.POST.get('vomiting')
        ques.Q0 = request.POST.get('Q0')
        ques.Q1 = request.POST.get('Q1')
        ques.Q2 = request.POST.get('Q2')
        ques.Q3 = request.POST.get('Q3')
        ques.Q4 = request.POST.get('Q4')
        ques.Q5 = request.POST.get('Q5')
        ques.Q6 = request.POST.get('Q6')
        ques.Q7 = request.POST.get('Q7')
        ques.Q8 = request.POST.get('Q8')
        ques.Q9 = request.POST.get('Q9')
        ques.Q10 = request.POST.get('Q10')
        ques.Q11 = request.POST.get('Q11')
        ques.Q12 = request.POST.get('Q12')
        ques.Q13 = request.POST.get('Q13')
        ques.Q14 = request.POST.get('Q14')
        ques.Q15 = request.POST.get('Q15')
        ques.Q16 = request.POST.get('Q16')
        ques.Q17 = request.POST.get('Q17')
        ques.Q18 = request.POST.get('Q18')
        ques.Q19 = request.POST.get('Q19')
        ques.Q20 = request.POST.get('Q20')
        ques.Q21 = request.POST.get('Q21')
        ques.Q22 = request.POST.get('Q22')
        ques.allergy = request.POST.get('allergy')
        ques.allergy1 = request.POST.get('allergy1')
        ques.allergy2 = request.POST.get('allergy2')
        ques.allergy3 = request.POST.get('allergy3')
        ques.allergy4 = request.POST.get('allergy4')
        ques.allergy5 = request.POST.get('allergy5')
        ques.Q23 = request.POST.get('Q23')
        ques.Q24 = request.POST.get('Q24')
        if len(request.FILES) != 0:
            prod.file = request.FILES['file']
        prod.save()
        side.save()
        ques.save()

        messages.success(request, "Successfully Submitted")
        return redirect('/')
    return render(request, 'Layout/LayoutIndex.html')


def preprocessing(data):
    # Palitan ng read something sql
    ds = pd.read_csv(data, encoding="unicode_escape")

    # Drop all column's NAs except NAs on symptoms
    ds = ds.dropna(subset=["SYMPTOM1", "SYMPTOM2",
                   "SYMPTOM3", "SYMPTOM4", "SYMPTOM5"])

    # Get all unique symptom
    symptoms = list(set([x for i in range(1, 6)
                    for x in ds[f"SYMPTOM{i}"].unique() if x is not np.nan]))

    # Combine all symptom column to one
    ds_copy_symptom = ds[["SYMPTOM1", "SYMPTOM2", "SYMPTOM3",
                          "SYMPTOM4", "SYMPTOM5"]].astype(str).agg(', '.join, axis=1)

    # Get important/target features
    ds = ds[["SEX", "ALLERGIES", "CUR_ILL",
             "HISTORY", "RECOVD", "VAX_NAME-Unique"]]

    # Replace values for recovered
    ds['SEX'] = ds['SEX'].apply(
        lambda x: 0 if x == "F" else 1 if x == "M" else 2 if x == "U" else None)

    # Replace values for allergies
    ds['ALLERGIES'] = ds['ALLERGIES'].apply(
        lambda x: 1 if x is not None else 0)

    # Replace values for current illness
    ds['CUR_ILL'] = ds['CUR_ILL'].apply(lambda x: 1 if x is not None else 0)

    # Replace values for recovered
    ds['HISTORY'] = ds['HISTORY'].apply(lambda x: 1 if x is not None else 0)

    # Replace values for recovered
    ds['RECOVD'] = ds['RECOVD'].apply(
        lambda x: "Yes" if x == "Y" else "No" if x == "N" else "U" if x == "NaN" else None)

    # Expand Symptom
    symptoms_df = pd.DataFrame()
    for symptom in symptoms:
        symptoms_df[symptom] = (ds_copy_symptom.str.contains(symptom))
    symptoms_df = symptoms_df.fillna(0)
    # symptoms.value_counts()
    print(symptoms_df.shape)
    print(ds.shape)

    # Combine symptoms and important features
    ds = pd.concat([ds, symptoms_df], axis=1)

    # ds = ds.dropna()

    # Discretize Categorical values
    colToLabel = []
    for col in ds.columns:
        if is_object_dtype(ds[col]):
            colToLabel.append(col)
    le = LabelEncoder()
    labels = {}
    for col in colToLabel:
        ds[col] = le.fit_transform(ds[col])
        labels[col] = dict(zip(le.classes_, range(len(le.classes_))))

    return ds

def build_model(data):
    ############# MODEL #################
    inertias = []  # wcss
    prev_y = 0
    highest = 0

    # Elbow method
    for x in range(1, 11):
        kmeans = KMeans(n_clusters=x, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        y = kmeans.inertia_
        inertias.append(y)
        print(y)

        if len(inertias) > 1:
            # Get the highest subsequent difference to be valued as k
            diff = abs(inertias[-2] - inertias[-1])
            if diff > highest:
                k = x
        # Get the highest subsequent difference to be valued as k

    # norm_distances = [float(i)/max(distances) for i in distances]
    # k = np.argmin(norm_distances) + 1

    # Set a final model with fixed k
    final_kmeans = KMeans(n_clusters=3, init='k-means++',
                          max_iter=300, n_init=10, random_state=0)

    return final_kmeans, inertias

def apply_scaler_pca(data):
    print(len(data))
    x_std = MinMaxScaler().fit_transform(data)

    pca = PCA(n_components=2)
    x_pca = pd.DataFrame(pca.fit_transform(x_std))
    print(len(x_pca))
    return x_pca


def checker_page(request):
    if request.POST:
        drop_header = request.POST.getlist('drop_header')
        print(drop_header)
        for head in drop_header:
            print(head)
        request.session['drop'] = drop_header
        method = request.POST.get('selected_method')
        if method == '1':
            return redirect('classification')
        elif method == '2':
            return redirect('clustering')
        else:
            return redirect('preprocessing')
    else:
        return render(request, 'system/index.html')


def chooseMethod(request):
    if request.method == 'POST':
        method = request.POST.get('method')
        print('method di session : ', method)
        request.session['method'] = method
    return redirect('classification')


def clustering(request):
    df = preprocessing("PreProcToAlgoVaers2020.csv")
    x_scaled = apply_scaler_pca(df)
    kmeans, inertia = build_model(x_scaled)

    # Menentukan kluster dari data
    kmeans.fit(x_scaled)

    # Menambahkan kolom "kluster" dalam data frame
    df['cluster'] = kmeans.labels_
    cluster = df['cluster'].value_counts()
    clusters = cluster.to_dict()
    sort_cluster = []
    label = []
    for i in sorted(clusters):
        sort_cluster.append(clusters[i])
        label.append(i)

    fig, ax = plt.subplots()
    sct = ax.scatter(x_scaled[0], x_scaled[1], s=200, c=df.cluster)
    legend1 = ax.legend(*sct.legend_elements(),
                        loc="lower left", title="Clusters")
    ax.add_artist(legend1)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200)
    plt.title("Clustering K-Means Results")
    plt.xlabel("Pca 1")
    plt.ylabel("Symptoms/pca 2")
    graph = get_graph()

# if name:
    data = {
        # "name": name,
        "clusters": sort_cluster,
        # "rows": rows,
        # "features": features,
        "label": label,
        "chart": graph,
    }

    return render(request, 'system/clustering.html', data)


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def services(request):
    return render(request, 'services.html')


def aboutUs(request):
    return render(request, 'about.html')


def contacts(request):
    return render(request, 'contact.html')


def verification(request):
    return render(request, 'Email_confirmation.html')


def home_page(request):
    return render(request, 'homepage.html')


def logout_view(request):
    logout(request)
    messages.success(request, "Successfully Logout!!")
    return redirect('/')


def viewDetails(request):
    userId = user.objects.get(author=request.user)
    sfId = sideeffect.objects.get(author=request.user)
    queId = questioner.objects.get(author=request.user)
    context = {
        'userId': userId,
        'sfId': sfId,
        'queId': queId
    }
    return render(request, 'view_details.html', context)

def login_page(request):
    # prod = get_object_or_404(user,id)
    if request.method == 'POST':
        user = auth.authenticate(
            username=request.POST['username'], password=request.POST['password'])
        if user is not None and user.is_superuser:
            auth.login(request, user)
            return redirect('/dashboard/')
        elif user is not None and user.is_active:
            auth.login(request, user)
            # id = User.objects.all().values_list('id', flat=True).filter(username=user)
            messages.success(request, 'Welcome to Vaccimo')
            time.sleep(1)
            return redirect('/vaccimo/')
        
            # return HttpResponse('/information-page/')
            # return redirect('/information-page/')
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('/loginpage/')
    else:
        return render(request, 'homepage.html')


def register_page(request):
    if request.method == 'POST':
        username = request.POST.get('username', None)
        email = request.POST.get('email', None)
        #first_name = request.POST.get('first_name', None)
        #last_name = request.POST.get('last_name', None)
        password = request.POST.get('password', None)
        confirm_password = request.POST.get('password_confirmation', None)

        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username is already taken')
                return redirect('/register/')
            elif User.objects.filter(email=email).exists():
                messages.error(request, 'Email is already taken')
                return redirect('/register/')
            else:
                user = User.objects.create_user(username=username, password=password, email=email)
                user.last_login = timezone.now() # Set last_login to current time
                user.save()
                auth.login(request, user)
                current_site = get_current_site(request)
                mail_subject = 'Activation link has been sent to your email id'
                message = render_to_string('acc_active_email.html', {
                    'user': user,
                    'domain': current_site.domain,
                    'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                    'token': account_activation_token.make_token(user),
                })
                email = EmailMessage(
                    mail_subject, message, to=[email]
                )
                email.send()
                return redirect('/email-verification/')
        else:
            return render(request, 'register.html', {'error': 'Both passwords are not matching'})

    else:
        return render(request, 'register.html')

def informationNew(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            if len(request.FILES) != 0:
                if len(prod.file) > 0:
                    os.remove(prod.file.path)
                prod.file = request.FILES['file']
            prod.nameFirst = request.POST.get('nameFirst')
            prod.nameLast = request.POST.get('nameLast')
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.asuffixge = request.POST.get('suffix')
            prod.gender = request.POST.get('gender')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst')
            prods.nameLast = request.POST.get('nameLast')
            prods.name = request.POST.get('name')
            prods.contact_number = request.POST.get('contact_number')
            prods.address = request.POST.get('address')
            prods.age = request.POST.get('age')
            prods.asuffixge = request.POST.get('suffix')
            prods.gender = request.POST.get('gender')
            if len(request.FILES) != 0:
                prods.files = request.FILES['file']
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/firstDose/')
        return render(request, 'newUserPage/information/information.html', {'prod': prod})

    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.nameFirst = request.POST.get('nameFirst')
            prod.nameLast = request.POST.get('nameLast')
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.asuffixge = request.POST.get('suffix')
            prod.gender = request.POST.get('gender')
            if len(request.FILES) != 0:
                prod.files = request.FILES['file']
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst')
            prods.nameLast = request.POST.get('nameLast')
            prods.name = request.POST.get('name')
            prods.contact_number = request.POST.get('contact_number')
            prods.address = request.POST.get('address')
            prods.age = request.POST.get('age')
            prods.asuffixge = request.POST.get('suffix')
            prods.gender = request.POST.get('gender')
            if len(request.FILES) != 0:
                prods.files = request.FILES['file']
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/firstDose/')
        return render(request, 'newUserPage/information/information.html')

 #1st Dose   
def firstDose(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.firstDose = request.POST.get('firstDose', 1)
            prod.dateVaccinated = request.POST.get('dateVaccinated')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst', prod.nameFirst)
            prods.nameLast = request.POST.get('nameLast', prod.nameLast)
            prods.name = request.POST.get('name', prod.name)
            prods.contact_number = request.POST.get('contact_number', prod.contact_number)
            prods.address = request.POST.get('address', prod.address)
            prods.age = request.POST.get('age', prod.age)
            prods.asuffixge = request.POST.get('suffix', prod.suffix)
            prods.gender = request.POST.get('gender', prod.gender)
            prods.vaccination_brand = request.POST.get('vaccination_brand', prod.vaccination_brand)
            prods.firstDose = request.POST.get('firstDose', prod.firstDose)
            prods.dateVaccinated = request.POST.get('dateVaccinated', prod.dateVaccinated)
            prods.vaccination_brand1 = request.POST.get('vaccination_brand1', prod.vaccination_brand1)
            prods.secondDose = request.POST.get('secondDose', prod.secondDose)
            prods.dateVaccinated1 = request.POST.get('dateVaccinated1', prod.dateVaccinated1)
            prods.vaccination_brand2 = request.POST.get('vaccination_brand2', prod.vaccination_brand2)
            prods.firstBooster = request.POST.get('firstBooster', prod.firstBooster)
            prods.dateVaccinated2 = request.POST.get('dateVaccinated2', prod.dateVaccinated2)
            prods.vaccination_brand3 = request.POST.get('vaccination_brand3', prod.vaccination_brand3)
            prods.secondBooster = request.POST.get('secondBooster', prod.secondBooster)
            prods.dateVaccinated3 = request.POST.get('dateVaccinated3', prod.dateVaccinated3)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/whatDoYouFeel/')
        return render(request, 'newUserPage/firstDose/index.html', {'prod': prod})
    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.firstDose = request.POST.get('firstDose', 1)
            prod.dateVaccinated = request.POST.get('dateVaccinated')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst', prod.nameFirst)
            prods.nameLast = request.POST.get('nameLast', prod.nameLast)
            prods.name = request.POST.get('name', prod.name)
            prods.contact_number = request.POST.get('contact_number', prod.contact_number)
            prods.address = request.POST.get('address', prod.address)
            prods.age = request.POST.get('age', prod.age)
            prods.asuffixge = request.POST.get('suffix', prod.suffix)
            prods.gender = request.POST.get('gender', prod.gender)
            prods.vaccination_brand = request.POST.get('vaccination_brand', prod.vaccination_brand)
            prods.firstDose = request.POST.get('firstDose', prod.firstDose)
            prods.dateVaccinated = request.POST.get('dateVaccinated', prod.dateVaccinated)
            prods.vaccination_brand1 = request.POST.get('vaccination_brand1', prod.vaccination_brand1)
            prods.secondDose = request.POST.get('secondDose', prod.secondDose)
            prods.dateVaccinated1 = request.POST.get('dateVaccinated1', prod.dateVaccinated1)
            prods.vaccination_brand2 = request.POST.get('vaccination_brand2', prod.vaccination_brand2)
            prods.firstBooster = request.POST.get('firstBooster', prod.firstBooster)
            prods.dateVaccinated2 = request.POST.get('dateVaccinated2', prod.dateVaccinated2)
            prods.vaccination_brand3 = request.POST.get('vaccination_brand3', prod.vaccination_brand3)
            prods.secondBooster = request.POST.get('secondBooster', prod.secondBooster)
            prods.dateVaccinated3 = request.POST.get('dateVaccinated3', prod.dateVaccinated3)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/whatDoYouFeel/')
        return render(request, 'newUserPage/firstDose/index.html')
    
#def followUp(request):
#    return render(request, 'newUserPage/followUp/index.html')

def surveyNew(request):
    try:
        prod = questioner.objects.get(author=request.user)
        if request.method == 'POST':
            # author_id = get
            # author = User.objects.get(id=author_id)
            prod.Q0 = request.POST.get('Q0')
            prod.Q1 = request.POST.get('Q1')
            prod.Q2 = request.POST.get('Q2')
            prod.Q3 = request.POST.get('Q3')
            prod.Q4 = request.POST.get('Q4')
            prod.Q5 = request.POST.get('Q5')
            prod.Q6 = request.POST.get('Q6')
            prod.Q7 = request.POST.get('Q7')
            prod.Q8 = request.POST.get('Q8')
            prod.Q9 = request.POST.get('Q9')
            prod.Q10 = request.POST.get('Q10')
            prod.Q11 = request.POST.get('Q11')
            prod.Q12 = request.POST.get('Q12')
            prod.Q13 = request.POST.get('Q13')
            prod.Q14 = request.POST.get('Q14')
            prod.Q15 = request.POST.get('Q15')
            prod.Q16 = request.POST.get('Q16')
            prod.Q17 = request.POST.get('Q17')
            prod.Q18 = request.POST.get('Q18')
            prod.Q19 = request.POST.get('Q19')
            prod.Q20 = request.POST.get('Q20')
            prod.Q21 = request.POST.get('Q21')
            prod.Q22 = request.POST.get('Q22')
            prod.allergy2 = request.POST.get('allergy2')
            prod.allergy3 = request.POST.get('allergy3')
            prod.allergy4 = request.POST.get('allergy4')
            prod.allergy5 = request.POST.get('allergy5')
            prod.Q23 = request.POST.get('Q23')
            prod.Q24 = request.POST.get('Q24')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/whatDoYouFeel/')
        else:
            users = user()
            return render(request, 'newUserPage/survey/index.html', {'prod': prod, 'users': users})

    except questioner.DoesNotExist:
        prod = questioner()
        if request.method == 'POST':
            prod.Q0 = request.POST.get('Q0')
            prod.Q1 = request.POST.get('Q1')
            prod.Q2 = request.POST.get('Q2')
            prod.Q3 = request.POST.get('Q3')
            prod.Q4 = request.POST.get('Q4')
            prod.Q5 = request.POST.get('Q5')
            prod.Q6 = request.POST.get('Q6')
            prod.Q7 = request.POST.get('Q7')
            prod.Q8 = request.POST.get('Q8')
            prod.Q9 = request.POST.get('Q9')
            prod.Q10 = request.POST.get('Q10')
            prod.Q11 = request.POST.get('Q11')
            prod.Q12 = request.POST.get('Q12')
            prod.Q13 = request.POST.get('Q13')
            prod.Q14 = request.POST.get('Q14')
            prod.Q15 = request.POST.get('Q15')
            prod.Q16 = request.POST.get('Q16')
            prod.Q17 = request.POST.get('Q17')
            prod.Q18 = request.POST.get('Q18')
            prod.Q19 = request.POST.get('Q19')
            prod.Q20 = request.POST.get('Q20')
            prod.Q21 = request.POST.get('Q21')
            prod.Q22 = request.POST.get('Q22')
            prod.allergy2 = request.POST.get('allergy2')
            prod.allergy3 = request.POST.get('allergy3')
            prod.allergy4 = request.POST.get('allergy4')
            prod.allergy5 = request.POST.get('allergy5')
            prod.Q23 = request.POST.get('Q23')
            prod.Q24 = request.POST.get('Q24')
            #    item = questioner(Q0=Q0,Q1=Q1,Q2=Q2,Q3=Q3,Q4=Q4,Q5=Q5,Q6=Q6,Q7=Q7,Q8=Q8,Q9=Q9,Q10=Q10,Q11=Q11,
            #     Q12=Q12,Q13=Q13,Q14=Q14,Q15=Q15,Q16=Q16,Q17=Q17,Q18=Q18,Q19=Q19,Q20=Q20,Q21=Q21,Q22=Q22,allergy=allergy,
            #     Q23=Q23,Q24=Q24)
            # instance = prod.save(commit=False)
            # instance.author = request.user
            # instance.save
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/whatDoYouFeel/')
        else:
            prod = user()
            return render(request, 'newUserPage/survey/index.html', {'prod': prod})
#question side effect
def whatDoYouFeel(request):
    try:
        prod = firstdose.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/side-effect-new/')
        else:
            return render(request, 'newUserPage/whatDoyouFeel/index.html', {'prod': prod})
    except firstdose.DoesNotExist:
        prod = firstdose()
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/side-effect-new/')
        else:
            return render(request, 'newUserPage/whatDoyouFeel/index.html') 
#sideeffect firstdose         
def sideEffectNew(request):
    try:    
        prod = firstdose.objects.get(author=request.user)
        if request.method == 'POST':
            prod.InjectionSitePain = request.POST.get('InjectionSitePain', 'No')
            prod.headache = request.POST.get('headache', 'No')
            prod.fever = request.POST.get('fever', 'No')
            prod.rashes = request.POST.get('rashes', 'No')
            prod.itchiness = request.POST.get('itchiness', 'No')
            prod.cough = request.POST.get('cough', 'No')
            prod.bodyPain = request.POST.get('bodyPain', 'No')
            prod.soarThroat = request.POST.get('soarThroat', 'No')
            prod.stomachAche = request.POST.get('stomachAche', 'No')
            prod.vomiting = request.POST.get('vomiting', 'No')
            prod.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', 'No')
            prod.chestPain = request.POST.get('chestPain', 'No')
            prod.disorentation = request.POST.get('disorentation', 'No')
            prod.tunnelVision = request.POST.get('tunnelVision', 'No')
            prod.seizure = request.POST.get('seizure', 'No')
            prod.others = request.POST.get('others', 'No')
            prod.author = request.user
            prod.save()
        prods = firstdoserestore() 
        if request.method == 'POST':
            prods.InjectionSitePain = request.POST.get('InjectionSitePain', prod.InjectionSitePain)
            prods.headache = request.POST.get('headache', prod.headache)
            prods.fever = request.POST.get('fever', prod.fever)
            prods.rashes = request.POST.get('rashes', prod.rashes)
            prods.itchiness = request.POST.get('itchiness', prod.itchiness)
            prods.cough = request.POST.get('cough', prod.cough)
            prods.bodyPain = request.POST.get('bodyPain', prod.bodyPain)
            prods.soarThroat = request.POST.get('soarThroat', prod.soarThroat)
            prods.stomachAche = request.POST.get('stomachAche', prod.stomachAche)
            prods.vomiting = request.POST.get('vomiting', prod.vomiting)
            prods.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', prod.difficultyOfBreathing)
            prods.chestPain = request.POST.get('chestPain', prod.chestPain)
            prods.disorentation = request.POST.get('disorentation', prod.disorentation)
            prods.tunnelVision = request.POST.get('tunnelVision', prod.tunnelVision)
            prods.seizure = request.POST.get('seizure', prod.seizure)
            prods.others = request.POST.get('others', prod.others)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/secondDoseQ/')
        else:
            return render(request, 'newUserPage/firstDose/sideEffect.html', {'prod': prod})
    except firstdose.DoesNotExist:
        prod = firstdose()
        if request.method == 'POST':
            prod.InjectionSitePain = request.POST.get('InjectionSitePain', 'No')
            prod.headache = request.POST.get('headache', 'No')
            prod.fever = request.POST.get('fever', 'No')
            prod.rashes = request.POST.get('rashes', 'No')
            prod.itchiness = request.POST.get('itchiness', 'No')
            prod.cough = request.POST.get('cough', 'No')
            prod.bodyPain = request.POST.get('bodyPain', 'No')
            prod.soarThroat = request.POST.get('soarThroat', 'No')
            prod.stomachAche = request.POST.get('stomachAche', 'No')
            prod.vomiting = request.POST.get('vomiting', 'No')
            prod.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', 'No')
            prod.chestPain = request.POST.get('chestPain', 'No')
            prod.disorentation = request.POST.get('disorentation', 'No')
            prod.tunnelVision = request.POST.get('tunnelVision', 'No')
            prod.seizure = request.POST.get('seizure', 'No')
            prod.others = request.POST.get('others', '')
            prod.author = request.user
            prod.save()
        prods = firstdoserestore()
        if request.method == 'POST':
            prods.InjectionSitePain = request.POST.get('InjectionSitePain', prod.InjectionSitePain)
            prods.headache = request.POST.get('headache', prod.headache)
            prods.fever = request.POST.get('fever', prod.fever)
            prods.rashes = request.POST.get('rashes', prod.rashes)
            prods.itchiness = request.POST.get('itchiness', prod.itchiness)
            prods.cough = request.POST.get('cough', prod.cough)
            prods.bodyPain = request.POST.get('bodyPain', prod.bodyPain)
            prods.soarThroat = request.POST.get('soarThroat', prod.soarThroat)
            prods.stomachAche = request.POST.get('stomachAche', prod.stomachAche)
            prods.vomiting = request.POST.get('vomiting', prod.vomiting)
            prods.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', prod.difficultyOfBreathing)
            prods.chestPain = request.POST.get('chestPain', prod.chestPain)
            prods.disorentation = request.POST.get('disorentation', prod.disorentation)
            prods.tunnelVision = request.POST.get('tunnelVision', prod.tunnelVision)
            prods.seizure = request.POST.get('seizure', prod.seizure)
            prods.others = request.POST.get('others', prod.others)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/secondDoseQ/')
        else:
            return render(request, 'newUserPage/firstDose/sideEffect.html')

               
#def whatDoYouFeel(request):
#    try:
#        prod = sideeffect.objects.get(author=request.user)
#        if request.method == 'POST'  and 'btnform1' in request.POST:
#            prod.name = request.POST.get('name')
#            prod.author = request.user
#            prod.save()
#            return redirect('/')
#        if request.method == 'POST'  and 'btnform2' in request.POST:
#            prod.name = request.POST.get('name')
#            prod.author = request.user
#            prod.save()
#            return redirect('/vaccimo/side-effect-new/')
#        else:
#            return render(request, 'newUserPage/whatDoyouFeel/index.html', {'prod': prod})
#    except sideeffect.DoesNotExist:
#        prod = sideeffect()
#        if request.method == 'POST'  and 'btnform1' in request.POST:
#            prod.name = request.POST.get('name')
#            prod.author = request.user
#            prod.save()
#            return redirect('/')
#        if request.method == 'POST'  and 'btnform2' in request.POST:
#            prod.name = request.POST.get('name')
#            prod.author = request.user
#            prod.save()
#            return redirect('/vaccimo/side-effect-new/')
#        else:
#            return render(request, 'newUserPage/whatDoyouFeel/index.html')
        
#def sideEffectNew(request):
#    try:    
#        prod = sideeffect.objects.get(author=request.user)
#        if request.method == 'POST':
#            prod.muscle_ache = request.POST.get('muscle_ache', 'No')
#            if prod.muscle_ache == "Yes":
#                prod.muscle_ache = 1
#            else:
#                prod.muscle_ache = 0
#            prod.headache = request.POST.get('headache', 'No')
#            if prod.headache == "Yes":
#                prod.headache = 1
#            else:
#                prod.headache = 0
#            prod.fever = request.POST.get('fever', 'No')
#            if prod.fever == "Yes":
#                prod.fever = 1
#            else:
#                prod.fever = 0
#            prod.redness = request.POST.get('redness', 'No')
#            if prod.redness == "Yes":
#                prod.redness = 1
#            else:
#                prod.redness = 0
#            prod.swelling = request.POST.get('swelling', 'No')
#            if prod.swelling == "Yes":
#                prod.swelling = 1
#            else:
#                prod.swelling = 0
#            prod.induration = request.POST.get('induration', 'No')
#            if prod.induration == "Yes":
#                prod.induration = 1
#            else:
#                prod.induration = 0
#            prod.chills = request.POST.get('chills', 'No')
#            if prod.chills == "Yes":
#                prod.chills = 1
#            else:
#                prod.chills = 0
#            prod.join_pain = request.POST.get('join_pain', 'No')
#            if prod.join_pain == "Yes":
#                prod.join_pain = 1
#            else:
#                prod.join_pain = 0
#            prod.fatigue = request.POST.get('fatigue', 'No')
#            if prod.fatigue == "Yes":
#                prod.fatigue = 1
#            else:
#                prod.fatigue = 0
#            prod.nausea = request.POST.get('nausea', 'No')
#            if prod.nausea == "Yes":
#                prod.nausea = 1
#            else:
#                prod.nausea = 0
#            prod.vomiting = request.POST.get('vomiting', 'No')
#            if prod.vomiting == "Yes":
#                prod.vomiting = 1
#            else:
#                prod.vomiting = 0

#        # Calculate total symptom value
#            total_symptoms = (prod.muscle_ache + prod.headache + prod.fever + prod.redness + prod.swelling + prod.induration + prod.chills + prod.join_pain + prod.fatigue + prod.nausea + prod.vomiting)
#            prod.feverish = total_symptoms
#            prod.author = request.user
#            prod.save()
            
#            if total_symptoms < 5:
#                severity = 'Mild'
#            elif total_symptoms <= 6:
#                severity = 'Moderate'
#            else:
#                severity = 'Severe'
#            messages.success(request, f"Successfully Submitted. Status: {severity}")
#            return redirect('/vaccimo/secondDoseQ/')
#        else:
#            return render(request, 'newUserPage/sideEffectNew/index.html', {'prod': prod})
#    except sideeffect.DoesNotExist:
#        prod = sideeffect()
#        if request.method == 'POST':
#            prod.muscle_ache = request.POST.get('muscle_ache')
#            if prod.muscle_ache == "Yes":
#                prod.muscle_ache = 1
#            else:
#                prod.muscle_ache = 0
#            prod.headache = request.POST.get('headache')
#            if prod.headache == "Yes":
#                prod.headache = 1
#            else:
#                prod.headache = 0
#            prod.fever = request.POST.get('fever')
#            if prod.fever == "Yes":
#                prod.fever = 1
#            else:
#                prod.fever = 0
#            prod.redness = request.POST.get('redness')
#            if prod.redness == "Yes":
#                prod.redness = 1
#            else:
#                prod.redness = 0
#            prod.swelling = request.POST.get('swelling')
#            if prod.swelling == "Yes":
#                prod.swelling = 1
#            else:
#                prod.swelling = 0
#            prod.induration = request.POST.get('induration')
#            if prod.induration == "Yes":
#                prod.induration = 1
#            else:
#                prod.induration = 0
#            prod.chills = request.POST.get('chills')
#            if prod.chills == "Yes":
#                prod.chills = 1
#            else:
#                prod.chills = 0
#            prod.join_pain = request.POST.get('join_pain')
#            if prod.join_pain == "Yes":
#                prod.join_pain = 1
#            else:
#                prod.join_pain = 0
#            prod.fatigue = request.POST.get('fatigue')
#            if prod.fatigue == "Yes":
#                prod.fatigue = 1
#            else:
#                prod.fatigue = 0
#            prod.nausea = request.POST.get('nausea')
#            if prod.nausea == "Yes":
#                prod.nausea = 1
#            else:
#                prod.nausea = 0
#            prod.vomiting = request.POST.get('vomiting')
#            if prod.vomiting == "Yes":
#                prod.vomiting = 1
#            else:
#                prod.vomiting = 0
#            # Calculate total symptom value
#            total_symptoms = (prod.muscle_ache + prod.headache + prod.fever + prod.redness + prod.swelling + prod.induration + prod.chills + prod.join_pain + prod.fatigue + prod.nausea + prod.vomiting)
#            prod.feverish = total_symptoms
#            prod.author = request.user
#            prod.save()
#            if total_symptoms < 5:
#                severity = 'Mild'
#            elif total_symptoms <= 6:
#                severity = 'Moderate'
#            else:
#                severity = 'Severe'
#            messages.success(request, f"Successfully Submitted. Status: {severity}")
#            return redirect('/vaccimo/secondDoseQ/')
#        else:
#            return render(request, 'newUserPage/sideEffectNew/index.html')
#2nd Dose
def secondDoseQ(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = seconddose.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            #prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            #prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/secondDose/')
        return render(request, 'newUserPage/secondDose/question.html', {'prod': prod})

    except seconddose.DoesNotExist:
        prod = seconddose()
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/secondDose/')
        return render(request, 'newUserPage/secondDose/question.html')

def secondDose(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            prod.vaccination_brand1 = request.POST.get('vaccination_brand1')
            prod.secondDose = request.POST.get('secondDose', 1)
            prod.dateVaccinated1 = request.POST.get('dateVaccinated1')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst', prod.nameFirst)
            prods.nameLast = request.POST.get('nameLast', prod.nameLast)
            prods.name = request.POST.get('name', prod.name)
            prods.contact_number = request.POST.get('contact_number', prod.contact_number)
            prods.address = request.POST.get('address', prod.address)
            prods.age = request.POST.get('age', prod.age)
            prods.asuffixge = request.POST.get('suffix', prod.suffix)
            prods.gender = request.POST.get('gender', prod.gender)
            prods.vaccination_brand = request.POST.get('vaccination_brand', prod.vaccination_brand)
            prods.firstDose = request.POST.get('firstDose', prod.firstDose)
            prods.dateVaccinated = request.POST.get('dateVaccinated', prod.dateVaccinated)
            prods.vaccination_brand1 = request.POST.get('vaccination_brand1', prod.vaccination_brand1)
            prods.secondDose = request.POST.get('secondDose', prod.secondDose)
            prods.dateVaccinated1 = request.POST.get('dateVaccinated1', prod.dateVaccinated1)
            prods.vaccination_brand2 = request.POST.get('vaccination_brand2', prod.vaccination_brand2)
            prods.firstBooster = request.POST.get('firstBooster', prod.firstBooster)
            prods.dateVaccinated2 = request.POST.get('dateVaccinated2', prod.dateVaccinated2)
            prods.vaccination_brand3 = request.POST.get('vaccination_brand3', prod.vaccination_brand3)
            prods.secondBooster = request.POST.get('secondBooster', prod.secondBooster)
            prods.dateVaccinated3 = request.POST.get('dateVaccinated3', prod.dateVaccinated3)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/whatDoYouFeel2/')
        return render(request, 'newUserPage/secondDose/index.html', {'prod': prod})

    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.vaccination_brand1 = request.POST.get('vaccination_brand1')
            prod.secondDose = request.POST.get('secondDose', 1)
            prod.dateVaccinated1 = request.POST.get('dateVaccinated1')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst', prod.nameFirst)
            prods.nameLast = request.POST.get('nameLast', prod.nameLast)
            prods.name = request.POST.get('name', prod.name)
            prods.contact_number = request.POST.get('contact_number', prod.contact_number)
            prods.address = request.POST.get('address', prod.address)
            prods.age = request.POST.get('age', prod.age)
            prods.asuffixge = request.POST.get('suffix', prod.suffix)
            prods.gender = request.POST.get('gender', prod.gender)
            prods.vaccination_brand = request.POST.get('vaccination_brand', prod.vaccination_brand)
            prods.firstDose = request.POST.get('firstDose', prod.firstDose)
            prods.dateVaccinated = request.POST.get('dateVaccinated', prod.dateVaccinated)
            prods.vaccination_brand1 = request.POST.get('vaccination_brand1', prod.vaccination_brand1)
            prods.secondDose = request.POST.get('secondDose', prod.secondDose)
            prods.dateVaccinated1 = request.POST.get('dateVaccinated1', prod.dateVaccinated1)
            prods.vaccination_brand2 = request.POST.get('vaccination_brand2', prod.vaccination_brand2)
            prods.firstBooster = request.POST.get('firstBooster', prod.firstBooster)
            prods.dateVaccinated2 = request.POST.get('dateVaccinated2', prod.dateVaccinated2)
            prods.vaccination_brand3 = request.POST.get('vaccination_brand3', prod.vaccination_brand3)
            prods.secondBooster = request.POST.get('secondBooster', prod.secondBooster)
            prods.dateVaccinated3 = request.POST.get('dateVaccinated3', prod.dateVaccinated3)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/whatDoYouFeel2/')
        return render(request, 'newUserPage/secondDose/index.html')

def whatDoYouFeel2(request):
    try:
        prod = seconddose.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/side-effect-new2/')
        else:
            return render(request, 'newUserPage/whatDoyouFeel/index2.html', {'prod': prod})
    except seconddose.DoesNotExist:
        prod = seconddose()
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/side-effect-new2/')
        else:
            return render(request, 'newUserPage/whatDoyouFeel/index2.html') 
            
def sideEffectNew2(request):
    try:    
        prod = seconddose.objects.get(author=request.user)
        if request.method == 'POST':
            prod.InjectionSitePain = request.POST.get('InjectionSitePain', 'No')
            prod.headache = request.POST.get('headache', 'No')
            prod.fever = request.POST.get('fever', 'No')
            prod.rashes = request.POST.get('rashes', 'No')
            prod.itchiness = request.POST.get('itchiness', 'No')
            prod.cough = request.POST.get('cough', 'No')
            prod.bodyPain = request.POST.get('bodyPain', 'No')
            prod.soarThroat = request.POST.get('soarThroat', 'No')
            prod.stomachAche = request.POST.get('stomachAche', 'No')
            prod.vomiting = request.POST.get('vomiting', 'No')
            prod.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', 'No')
            prod.chestPain = request.POST.get('chestPain', 'No')
            prod.disorentation = request.POST.get('disorentation', 'No')
            prod.tunnelVision = request.POST.get('tunnelVision', 'No')
            prod.seizure = request.POST.get('seizure', 'No')
            prod.others = request.POST.get('others', '')
            prod.author = request.user
            prod.save()
        prods = seconddoserestore()
        if request.method == 'POST':
            prods.InjectionSitePain = request.POST.get('InjectionSitePain', prod.InjectionSitePain)
            prods.headache = request.POST.get('headache', prod.headache)
            prods.fever = request.POST.get('fever', prod.fever)
            prods.rashes = request.POST.get('rashes', prod.rashes)
            prods.itchiness = request.POST.get('itchiness', prod.itchiness)
            prods.cough = request.POST.get('cough', prod.cough)
            prods.bodyPain = request.POST.get('bodyPain', prod.bodyPain)
            prods.soarThroat = request.POST.get('soarThroat', prod.soarThroat)
            prods.stomachAche = request.POST.get('stomachAche', prod.stomachAche)
            prods.vomiting = request.POST.get('vomiting', prod.vomiting)
            prods.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', prod.difficultyOfBreathing)
            prods.chestPain = request.POST.get('chestPain', prod.chestPain)
            prods.disorentation = request.POST.get('disorentation', prod.disorentation)
            prods.tunnelVision = request.POST.get('tunnelVision', prod.tunnelVision)
            prods.seizure = request.POST.get('seizure', prod.seizure)
            prods.others = request.POST.get('others', prod.others)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/firstBoosterQuestion/')
        else:
            return render(request, 'newUserPage/twoSideEffect/index.html', {'prod': prod})
    except seconddose.DoesNotExist:
        prod = seconddose()
        if request.method == 'POST':
            prod.InjectionSitePain = request.POST.get('InjectionSitePain', 'No')
            prod.headache = request.POST.get('headache', 'No')
            prod.fever = request.POST.get('fever', 'No')
            prod.rashes = request.POST.get('rashes', 'No')
            prod.itchiness = request.POST.get('itchiness', 'No')
            prod.cough = request.POST.get('cough', 'No')
            prod.bodyPain = request.POST.get('bodyPain', 'No')
            prod.soarThroat = request.POST.get('soarThroat', 'No')
            prod.stomachAche = request.POST.get('stomachAche', 'No')
            prod.vomiting = request.POST.get('vomiting', 'No')
            prod.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', 'No')
            prod.chestPain = request.POST.get('chestPain', 'No')
            prod.disorentation = request.POST.get('disorentation', 'No')
            prod.tunnelVision = request.POST.get('tunnelVision', 'No')
            prod.seizure = request.POST.get('seizure', 'No')
            prod.others = request.POST.get('others', '')
            prod.author = request.user
            prod.save()
        prods = seconddoserestore()
        if request.method == 'POST':
            prods.InjectionSitePain = request.POST.get('InjectionSitePain', prod.InjectionSitePain)
            prods.headache = request.POST.get('headache', prod.headache)
            prods.fever = request.POST.get('fever', prod.fever)
            prods.rashes = request.POST.get('rashes', prod.rashes)
            prods.itchiness = request.POST.get('itchiness', prod.itchiness)
            prods.cough = request.POST.get('cough', prod.cough)
            prods.bodyPain = request.POST.get('bodyPain', prod.bodyPain)
            prods.soarThroat = request.POST.get('soarThroat', prod.soarThroat)
            prods.stomachAche = request.POST.get('stomachAche', prod.stomachAche)
            prods.vomiting = request.POST.get('vomiting', prod.vomiting)
            prods.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', prod.difficultyOfBreathing)
            prods.chestPain = request.POST.get('chestPain', prod.chestPain)
            prods.disorentation = request.POST.get('disorentation', prod.disorentation)
            prods.tunnelVision = request.POST.get('tunnelVision', prod.tunnelVision)
            prods.seizure = request.POST.get('seizure', prod.seizure)
            prods.others = request.POST.get('others', prod.others)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/firstBoosterQuestion/')
        else:
            return render(request, 'newUserPage/twoSideEffect/index.html')
#1st booster
def firstBoosterQuestion(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/firstBooster/')
        return render(request, 'newUserPage/firstBooster/question.html', {'prod': prod})

    except user.DoesNotExist:
        prod = user.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/firstBooster/')
        return render(request, 'newUserPage/firstBooster/question.html')
    
def firstBooster(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            prod.vaccination_brand2 = request.POST.get('vaccination_brand2')
            prod.firstBooster = request.POST.get('firstBooster', 1)
            prod.dateVaccinated2 = request.POST.get('dateVaccinated2')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst', prod.nameFirst)
            prods.nameLast = request.POST.get('nameLast', prod.nameLast)
            prods.name = request.POST.get('name', prod.name)
            prods.contact_number = request.POST.get('contact_number', prod.contact_number)
            prods.address = request.POST.get('address', prod.address)
            prods.age = request.POST.get('age', prod.age)
            prods.asuffixge = request.POST.get('suffix', prod.suffix)
            prods.gender = request.POST.get('gender', prod.gender)
            prods.vaccination_brand = request.POST.get('vaccination_brand', prod.vaccination_brand)
            prods.firstDose = request.POST.get('firstDose', prod.firstDose)
            prods.dateVaccinated = request.POST.get('dateVaccinated', prod.dateVaccinated)
            prods.vaccination_brand1 = request.POST.get('vaccination_brand1', prod.vaccination_brand1)
            prods.secondDose = request.POST.get('secondDose', prod.secondDose)
            prods.dateVaccinated1 = request.POST.get('dateVaccinated1', prod.dateVaccinated1)
            prods.vaccination_brand2 = request.POST.get('vaccination_brand2', prod.vaccination_brand2)
            prods.firstBooster = request.POST.get('firstBooster', prod.firstBooster)
            prods.dateVaccinated2 = request.POST.get('dateVaccinated2', prod.dateVaccinated2)
            prods.vaccination_brand3 = request.POST.get('vaccination_brand3', prod.vaccination_brand3)
            prods.secondBooster = request.POST.get('secondBooster', prod.secondBooster)
            prods.dateVaccinated3 = request.POST.get('dateVaccinated3', prod.dateVaccinated3)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/whatDoYouFeel3/')
        return render(request, 'newUserPage/firstBooster/index.html', {'prod': prod})

    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.vaccination_brand2 = request.POST.get('vaccination_brand2')
            prod.firstBooster = request.POST.get('firstBooster', 1)
            prod.dateVaccinated2 = request.POST.get('dateVaccinated2')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst', prod.nameFirst)
            prods.nameLast = request.POST.get('nameLast', prod.nameLast)
            prods.name = request.POST.get('name', prod.name)
            prods.contact_number = request.POST.get('contact_number', prod.contact_number)
            prods.address = request.POST.get('address', prod.address)
            prods.age = request.POST.get('age', prod.age)
            prods.asuffixge = request.POST.get('suffix', prod.suffix)
            prods.gender = request.POST.get('gender', prod.gender)
            prods.vaccination_brand = request.POST.get('vaccination_brand', prod.vaccination_brand)
            prods.firstDose = request.POST.get('firstDose', prod.firstDose)
            prods.dateVaccinated = request.POST.get('dateVaccinated', prod.dateVaccinated)
            prods.vaccination_brand1 = request.POST.get('vaccination_brand1', prod.vaccination_brand1)
            prods.secondDose = request.POST.get('secondDose', prod.secondDose)
            prods.dateVaccinated1 = request.POST.get('dateVaccinated1', prod.dateVaccinated1)
            prods.vaccination_brand2 = request.POST.get('vaccination_brand2', prod.vaccination_brand2)
            prods.firstBooster = request.POST.get('firstBooster', prod.firstBooster)
            prods.dateVaccinated2 = request.POST.get('dateVaccinated2', prod.dateVaccinated2)
            prods.vaccination_brand3 = request.POST.get('vaccination_brand3', prod.vaccination_brand3)
            prods.secondBooster = request.POST.get('secondBooster', prod.secondBooster)
            prods.dateVaccinated3 = request.POST.get('dateVaccinated3', prod.dateVaccinated3)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/whatDoYouFeel3/')
        return render(request, 'newUserPage/firstBooster/index.html')

def whatDoYouFeel3(request):
    try:
        prod = firstbooster.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/side-effect-new3/')
        else:
            return render(request, 'newUserPage/whatDoyouFeel/index3.html', {'prod': prod})
    except firstbooster.DoesNotExist:
        prod = firstbooster()
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/side-effect-new3/')
        else:
            return render(request, 'newUserPage/whatDoyouFeel/index3.html')
            
def sideEffectNew3(request):
    try:    
        prod = firstbooster.objects.get(author=request.user)
        if request.method == 'POST':
            prod.InjectionSitePain = request.POST.get('InjectionSitePain', 'No')
            prod.headache = request.POST.get('headache', 'No')
            prod.fever = request.POST.get('fever', 'No')
            prod.rashes = request.POST.get('rashes', 'No')
            prod.itchiness = request.POST.get('itchiness', 'No')
            prod.cough = request.POST.get('cough', 'No')
            prod.bodyPain = request.POST.get('bodyPain', 'No')
            prod.soarThroat = request.POST.get('soarThroat', 'No')
            prod.stomachAche = request.POST.get('stomachAche', 'No')
            prod.vomiting = request.POST.get('vomiting', 'No')
            prod.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', 'No')
            prod.chestPain = request.POST.get('chestPain', 'No')
            prod.disorentation = request.POST.get('disorentation', 'No')
            prod.tunnelVision = request.POST.get('tunnelVision', 'No')
            prod.seizure = request.POST.get('seizure', 'No')
            prod.others = request.POST.get('others', '')
            prod.author = request.user
            prod.save()
        prods = firstboosterrestore()
        if request.method == 'POST':
            prods.InjectionSitePain = request.POST.get('InjectionSitePain', prod.InjectionSitePain)
            prods.headache = request.POST.get('headache', prod.headache)
            prods.fever = request.POST.get('fever', prod.fever)
            prods.rashes = request.POST.get('rashes', prod.rashes)
            prods.itchiness = request.POST.get('itchiness', prod.itchiness)
            prods.cough = request.POST.get('cough', prod.cough)
            prods.bodyPain = request.POST.get('bodyPain', prod.bodyPain)
            prods.soarThroat = request.POST.get('soarThroat', prod.soarThroat)
            prods.stomachAche = request.POST.get('stomachAche', prod.stomachAche)
            prods.vomiting = request.POST.get('vomiting', prod.vomiting)
            prods.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', prod.difficultyOfBreathing)
            prods.chestPain = request.POST.get('chestPain', prod.chestPain)
            prods.disorentation = request.POST.get('disorentation', prod.disorentation)
            prods.tunnelVision = request.POST.get('tunnelVision', prod.tunnelVision)
            prods.seizure = request.POST.get('seizure', prod.seizure)
            prods.others = request.POST.get('others', prod.others)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/secondBoosterQuestion/')
        else:
            return render(request, 'newUserPage/twoSideEffect/index.html', {'prod': prod})
    except firstbooster.DoesNotExist:
        prod = firstbooster()
        if request.method == 'POST':
            prod.InjectionSitePain = request.POST.get('InjectionSitePain', 'No')
            prod.headache = request.POST.get('headache', 'No')
            prod.fever = request.POST.get('fever', 'No')
            prod.rashes = request.POST.get('rashes', 'No')
            prod.itchiness = request.POST.get('itchiness', 'No')
            prod.cough = request.POST.get('cough', 'No')
            prod.bodyPain = request.POST.get('bodyPain', 'No')
            prod.soarThroat = request.POST.get('soarThroat', 'No')
            prod.stomachAche = request.POST.get('stomachAche', 'No')
            prod.vomiting = request.POST.get('vomiting', 'No')
            prod.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', 'No')
            prod.chestPain = request.POST.get('chestPain', 'No')
            prod.disorentation = request.POST.get('disorentation', 'No')
            prod.tunnelVision = request.POST.get('tunnelVision', 'No')
            prod.seizure = request.POST.get('seizure', 'No')
            prod.others = request.POST.get('others', '')
        prods = firstboosterrestore()
        if request.method == 'POST':
            prods.InjectionSitePain = request.POST.get('InjectionSitePain', prod.InjectionSitePain)
            prods.headache = request.POST.get('headache', prod.headache)
            prods.fever = request.POST.get('fever', prod.fever)
            prods.rashes = request.POST.get('rashes', prod.rashes)
            prods.itchiness = request.POST.get('itchiness', prod.itchiness)
            prods.cough = request.POST.get('cough', prod.cough)
            prods.bodyPain = request.POST.get('bodyPain', prod.bodyPain)
            prods.soarThroat = request.POST.get('soarThroat', prod.soarThroat)
            prods.stomachAche = request.POST.get('stomachAche', prod.stomachAche)
            prods.vomiting = request.POST.get('vomiting', prod.vomiting)
            prods.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', prod.difficultyOfBreathing)
            prods.chestPain = request.POST.get('chestPain', prod.chestPain)
            prods.disorentation = request.POST.get('disorentation', prod.disorentation)
            prods.tunnelVision = request.POST.get('tunnelVision', prod.tunnelVision)
            prods.seizure = request.POST.get('seizure', prod.seizure)
            prods.others = request.POST.get('others', prod.others)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/secondBoosterQuestion/')
        else:
            return render(request, 'newUserPage/twoSideEffect/index.html')

#2nd booster
def secondBoosterQuestion(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/secondBooster/')
        return render(request, 'newUserPage/secondBooster/question.html', {'prod': prod})

    except user.DoesNotExist:
        prod = user.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/secondBooster/')
        return render(request, 'newUserPage/secondBooster/question.html')
    
def secondBooster(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            prod.vaccination_brand3 = request.POST.get('vaccination_brand3')
            prod.secondBooster = request.POST.get('secondBooster', 1)
            prod.dateVaccinated3 = request.POST.get('dateVaccinated3')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst', prod.nameFirst)
            prods.nameLast = request.POST.get('nameLast', prod.nameLast)
            prods.name = request.POST.get('name', prod.name)
            prods.contact_number = request.POST.get('contact_number', prod.contact_number)
            prods.address = request.POST.get('address', prod.address)
            prods.age = request.POST.get('age', prod.age)
            prods.asuffixge = request.POST.get('suffix', prod.suffix)
            prods.gender = request.POST.get('gender', prod.gender)
            prods.vaccination_brand = request.POST.get('vaccination_brand', prod.vaccination_brand)
            prods.firstDose = request.POST.get('firstDose', prod.firstDose)
            prods.dateVaccinated = request.POST.get('dateVaccinated', prod.dateVaccinated)
            prods.vaccination_brand1 = request.POST.get('vaccination_brand1', prod.vaccination_brand1)
            prods.secondDose = request.POST.get('secondDose', prod.secondDose)
            prods.dateVaccinated1 = request.POST.get('dateVaccinated1', prod.dateVaccinated1)
            prods.vaccination_brand2 = request.POST.get('vaccination_brand2', prod.vaccination_brand2)
            prods.firstBooster = request.POST.get('firstBooster', prod.firstBooster)
            prods.dateVaccinated2 = request.POST.get('dateVaccinated2', prod.dateVaccinated2)
            prods.vaccination_brand3 = request.POST.get('vaccination_brand3', prod.vaccination_brand3)
            prods.secondBooster = request.POST.get('secondBooster', prod.secondBooster)
            prods.dateVaccinated3 = request.POST.get('dateVaccinated3', prod.dateVaccinated3)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/whatDoYouFeel4/')
        return render(request, 'newUserPage/secondBooster/index.html', {'prod': prod})

    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.vaccination_brand3 = request.POST.get('vaccination_brand3')
            prod.secondBooster = request.POST.get('secondBooster', 1)
            prod.dateVaccinated3 = request.POST.get('dateVaccinated3')
            prod.author = request.user
            prod.save()
        prods = userRestorData()
        if request.method == 'POST':
            prods.nameFirst = request.POST.get('nameFirst', prod.nameFirst)
            prods.nameLast = request.POST.get('nameLast', prod.nameLast)
            prods.name = request.POST.get('name', prod.name)
            prods.contact_number = request.POST.get('contact_number', prod.contact_number)
            prods.address = request.POST.get('address', prod.address)
            prods.age = request.POST.get('age', prod.age)
            prods.asuffixge = request.POST.get('suffix', prod.suffix)
            prods.gender = request.POST.get('gender', prod.gender)
            prods.vaccination_brand = request.POST.get('vaccination_brand', prod.vaccination_brand)
            prods.firstDose = request.POST.get('firstDose', prod.firstDose)
            prods.dateVaccinated = request.POST.get('dateVaccinated', prod.dateVaccinated)
            prods.vaccination_brand1 = request.POST.get('vaccination_brand1', prod.vaccination_brand1)
            prods.secondDose = request.POST.get('secondDose', prod.secondDose)
            prods.dateVaccinated1 = request.POST.get('dateVaccinated1', prod.dateVaccinated1)
            prods.vaccination_brand2 = request.POST.get('vaccination_brand2', prod.vaccination_brand2)
            prods.firstBooster = request.POST.get('firstBooster', prod.firstBooster)
            prods.dateVaccinated2 = request.POST.get('dateVaccinated2', prod.dateVaccinated2)
            prods.vaccination_brand3 = request.POST.get('vaccination_brand3', prod.vaccination_brand3)
            prods.secondBooster = request.POST.get('secondBooster', prod.secondBooster)
            prods.dateVaccinated3 = request.POST.get('dateVaccinated3', prod.dateVaccinated3)
            prods.author = request.user
            prods.save()
            return redirect('/vaccimo/whatDoYouFeel4/')
        return render(request, 'newUserPage/secondBooster/index.html')
    
def whatDoYouFeel4(request):
    try:
        prod = secondbooster.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/side-effect-new4/')
        else:
            return render(request, 'newUserPage/whatDoyouFeel/index4.html', {'prod': prod})
    except secondbooster.DoesNotExist:
        prod = secondbooster()
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.user = request.POST.get('user')
            prod.author = request.user
            prod.save()
            return redirect('/vaccimo/side-effect-new4/')
        else:
            return render(request, 'newUserPage/whatDoyouFeel/index4.html')
            
def sideEffectNew4(request):
    try:    
        prod = secondbooster.objects.get(author=request.user)
        if request.method == 'POST':
            prod.InjectionSitePain = request.POST.get('InjectionSitePain', 'No')
            prod.headache = request.POST.get('headache', 'No')
            prod.fever = request.POST.get('fever', 'No')
            prod.rashes = request.POST.get('rashes', 'No')
            prod.itchiness = request.POST.get('itchiness', 'No')
            prod.cough = request.POST.get('cough', 'No')
            prod.bodyPain = request.POST.get('bodyPain', 'No')
            prod.soarThroat = request.POST.get('soarThroat', 'No')
            prod.stomachAche = request.POST.get('stomachAche', 'No')
            prod.vomiting = request.POST.get('vomiting', 'No')
            prod.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', 'No')
            prod.chestPain = request.POST.get('chestPain', 'No')
            prod.disorentation = request.POST.get('disorentation', 'No')
            prod.tunnelVision = request.POST.get('tunnelVision', 'No')
            prod.seizure = request.POST.get('seizure', 'No')
            prod.others = request.POST.get('others', '')
            prod.author = request.user
            prod.save()
        prods = secondboosterrestored()
        if request.method == 'POST':
            prods.InjectionSitePain = request.POST.get('InjectionSitePain', prod.InjectionSitePain)
            prods.headache = request.POST.get('headache', prod.headache)
            prods.fever = request.POST.get('fever', prod.fever)
            prods.rashes = request.POST.get('rashes', prod.rashes)
            prods.itchiness = request.POST.get('itchiness', prod.itchiness)
            prods.cough = request.POST.get('cough', prod.cough)
            prods.bodyPain = request.POST.get('bodyPain', prod.bodyPain)
            prods.soarThroat = request.POST.get('soarThroat', prod.soarThroat)
            prods.stomachAche = request.POST.get('stomachAche', prod.stomachAche)
            prods.vomiting = request.POST.get('vomiting', prod.vomiting)
            prods.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', prod.difficultyOfBreathing)
            prods.chestPain = request.POST.get('chestPain', prod.chestPain)
            prods.disorentation = request.POST.get('disorentation', prod.disorentation)
            prods.tunnelVision = request.POST.get('tunnelVision', prod.tunnelVision)
            prods.seizure = request.POST.get('seizure', prod.seizure)
            prods.others = request.POST.get('others', prod.others)
            prods.author = request.user
            prods.save()
            messages.success(request, 'Successfully Submited!!!')
            return redirect('/')
        else:
            return render(request, 'newUserPage/twoSideEffect/index.html', {'prod': prod})
    except secondbooster.DoesNotExist:
        prod = secondbooster()
        if request.method == 'POST':
            prod.InjectionSitePain = request.POST.get('InjectionSitePain', 'No')
            prod.headache = request.POST.get('headache', 'No')
            prod.fever = request.POST.get('fever', 'No')
            prod.rashes = request.POST.get('rashes', 'No')
            prod.itchiness = request.POST.get('itchiness', 'No')
            prod.cough = request.POST.get('cough', 'No')
            prod.bodyPain = request.POST.get('bodyPain', 'No')
            prod.soarThroat = request.POST.get('soarThroat', 'No')
            prod.stomachAche = request.POST.get('stomachAche', 'No')
            prod.vomiting = request.POST.get('vomiting', 'No')
            prod.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', 'No')
            prod.chestPain = request.POST.get('chestPain', 'No')
            prod.disorentation = request.POST.get('disorentation', 'No')
            prod.tunnelVision = request.POST.get('tunnelVision', 'No')
            prod.seizure = request.POST.get('seizure', 'No')
            prod.others = request.POST.get('others', '')
            prod.author = request.user
        prods = secondboosterrestored()
        if request.method == 'POST':
            prods.InjectionSitePain = request.POST.get('InjectionSitePain', prod.InjectionSitePain)
            prods.headache = request.POST.get('headache', prod.headache)
            prods.fever = request.POST.get('fever', prod.fever)
            prods.rashes = request.POST.get('rashes', prod.rashes)
            prods.itchiness = request.POST.get('itchiness', prod.itchiness)
            prods.cough = request.POST.get('cough', prod.cough)
            prods.bodyPain = request.POST.get('bodyPain', prod.bodyPain)
            prods.soarThroat = request.POST.get('soarThroat', prod.soarThroat)
            prods.stomachAche = request.POST.get('stomachAche', prod.stomachAche)
            prods.vomiting = request.POST.get('vomiting', prod.vomiting)
            prods.difficultyOfBreathing = request.POST.get('difficultyOfBreathing', prod.difficultyOfBreathing)
            prods.chestPain = request.POST.get('chestPain', prod.chestPain)
            prods.disorentation = request.POST.get('disorentation', prod.disorentation)
            prods.tunnelVision = request.POST.get('tunnelVision', prod.tunnelVision)
            prods.seizure = request.POST.get('seizure', prod.seizure)
            prods.others = request.POST.get('others', prod.others)
            prod.save()
            return redirect('/')
        else:
            return render(request, 'newUserPage/twoSideEffect/index.html')                     
def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        messages.success(request, 'Welcome to Vaccimo')
        return redirect('/vaccimo/')
    else:
        return HttpResponse('Activation link is invalid!')

def information_page(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            if len(request.FILES) != 0:
                if len(prod.file) > 0:
                    os.remove(prod.file.path)
                prod.file = request.FILES['file']
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            prod.author = request.user
            prod.save()
            return redirect('/server-form-page/')
        return render(request, 'information.html', {'prod': prod})
    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            if len(request.FILES) != 0:
                prod.files = request.FILES['file']
            prod.author = request.user
            prod.save()
            return redirect('/server-form-page/')
        return render(request, 'information.html')
#def information_page(request):
#    prod = user()
#    if request.method == 'POST':
#        prod.name = request.POST.get('name')
#        prod.contact_number = request.POST.get('contact_number')
#        prod.vaccination_brand = request.POST.get('vaccination_brand')
#        prod.vaccination_site = request.POST.get('vaccination_site')
#        prod.address = request.POST.get('address')
#        prod.age = request.POST.get('age')
#        prod.bday = request.POST.get('bday')
#        prod.gender = request.POST.get('gender')
#        if len(request.FILES) != 0:
#            prod.file = request.FILES['file']
#        prod.author = request.user
#        prod.save()
#        return redirect('/server-form-page/')
#    return render(request, 'information.html')
    # prod = user()
    # if request.method == 'POST':
    #     prod.name = request.POST.get('name')
    #     prod.contact_number = request.POST.get('contact_number')
    #     prod.vaccination_brand = request.POST.get('vaccination_brand')
    #     prod.vaccination_site = request.POST.get('vaccination_site')
    #     prod.address = request.POST.get('address')
    #     prod.age = request.POST.get('age')
    #     prod.bday = request.POST.get('bday')
    #     prod.gender = request.POST.get('gender')
    #     if len(request.FILES) != 0:
    #         prod.file = request.FILES['file']
    #     prod.author = request.user.id
    #     prod.save()
    #     messages.success(request, 'Successfully Submitted')
    #     return redirect('/')
    # return render(request, 'information.html')


def profileEdit(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            if len(request.FILES) != 0:
                if len(prod.file) > 0:
                    os.remove(prod.file.path)
                prod.file = request.FILES['file']
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            prod.author = request.user
            prod.save()
            return redirect('/server-form-page/')
        return render(request, 'profileEdit.html', {'prod': prod})
    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            if len(request.FILES) != 0:
                prod.files = request.FILES['file']
            prod.author = request.user
            prod.save()
            return redirect('/server-form-page/')
        return render(request, 'profileEdit.html')

        # prod = user.objects.get(pk=request.user.id)
        # if request.method == 'POST':
        #     if len(request.FILES) !=0:
        #         if len(prod.file) >0:
        #             os.remove(prod.file.path)
        #         prod.file = request.FILES['file']
        #     prod.name = request.POST.get('name')
        #     prod.contact_number= request.POST.get('contact_number')
        #     prod.vaccination_brand = request.POST.get('vaccination_brand')
        #     prod.vaccination_site = request.POST.get('vaccination_site')
        #     prod.address = request.POST.get('address')
        #     prod.age = request.POST.get('age')
        #     prod.bday = request.POST.get('bday')
        #     prod.gender = request.POST.get('gender')
        #     prod.save()
        #     return redirect('/server-form-page/')
        # return render (request,'profileEdit.html',{'prod':prod})


# def information_page(request):
#     if request.method == 'POST':
#          prod = user()
#          prod.name = request.POST.get('name')
#          prod.contact_number= request.POST.get('contact_number')
#          prod.vaccination_brand = request.POST.get('vaccination_brand')
#          prod.vaccination_site = request.POST.get('vaccination_site')
#          prod.address = request.POST.get('address')
#          prod.age = request.POST.get('age')
#          prod.bday = request.POST.get('bday')
#          prod.gender = request.POST.get('gender')
#          if len(request.FILES) !=0:
#              prod.file = request.FILES['file']
#          prod.save()

#          return redirect('server')
#     return render (request,'information.html')


def sideeffect_page(request):
    try:    
        prod = sideeffect.objects.get(author=request.user)
        if request.method == 'POST':
            prod.muscle_ache = request.POST.get('muscle_ache')
            prod.headache = request.POST.get('headache')
            prod.fever = request.POST.get('fever')
            prod.redness = request.POST.get('redness')
            prod.swelling = request.POST.get('swelling')
            prod.tenderness = request.POST.get('tenderness')
            prod.warmth = request.POST.get('warmth')
            prod.itch = request.POST.get('itch')
            prod.induration = request.POST.get('induration')
            prod.feverish = request.POST.get('feverish')
            prod.chills = request.POST.get('chills')
            prod.join_pain = request.POST.get('join_pain')
            prod.fatigue = request.POST.get('fatigue')
            prod.nausea = request.POST.get('nausea')
            prod.vomiting = request.POST.get('vomiting')
            prod.author = request.user
            prod.save()
            messages.success(request, "Successfully Submitted")
            return redirect('/')
            # return render (request,'login_page.html')
        else:
            return render(request, 'sideeffect.html', {'prod': prod})
    except sideeffect.DoesNotExist:
        prod = sideeffect()
        if request.method == 'POST':
            prod.muscle_ache = request.POST.get('muscle_ache')
            prod.headache = request.POST.get('headache')
            prod.fever = request.POST.get('fever')
            prod.redness = request.POST.get('redness')
            prod.swelling = request.POST.get('swelling')
            prod.tenderness = request.POST.get('tenderness')
            prod.warmth = request.POST.get('warmth')
            prod.itch = request.POST.get('itch')
            prod.induration = request.POST.get('induration')
            prod.feverish = request.POST.get('feverish')
            prod.chills = request.POST.get('chills')
            prod.join_pain = request.POST.get('join_pain')
            prod.fatigue = request.POST.get('fatigue')
            prod.nausea = request.POST.get('nausea')
            prod.vomiting = request.POST.get('vomiting')
            prod.author = request.user
            prod.save()
            messages.success(request, "Successfully Submitted")
            return redirect('/')
        else:
            return render(request, 'sideeffect.html')


def server_form(request):
    try:
        prod = questioner.objects.get(author=request.user)
        if request.method == 'POST':
            # author_id = get
            # author = User.objects.get(id=author_id)
            prod.Q0 = request.POST.get('Q0')
            prod.Q1 = request.POST.get('Q1')
            prod.Q2 = request.POST.get('Q2')
            prod.Q3 = request.POST.get('Q3')
            prod.Q4 = request.POST.get('Q4')
            prod.Q5 = request.POST.get('Q5')
            prod.Q6 = request.POST.get('Q6')
            prod.Q7 = request.POST.get('Q7')
            prod.Q8 = request.POST.get('Q8')
            prod.Q9 = request.POST.get('Q9')
            prod.Q10 = request.POST.get('Q10')
            prod.Q11 = request.POST.get('Q11')
            prod.Q12 = request.POST.get('Q12')
            prod.Q13 = request.POST.get('Q13')
            prod.Q14 = request.POST.get('Q14')
            prod.Q15 = request.POST.get('Q15')
            prod.Q16 = request.POST.get('Q16')
            prod.Q17 = request.POST.get('Q17')
            prod.Q18 = request.POST.get('Q18')
            prod.Q19 = request.POST.get('Q19')
            prod.Q20 = request.POST.get('Q20')
            prod.Q21 = request.POST.get('Q21')
            prod.Q22 = request.POST.get('Q22')
            prod.allergy = request.POST.get('allergy')
            prod.allergy1 = request.POST.get('allergy1')
            prod.allergy2 = request.POST.get('allergy2')
            prod.allergy3 = request.POST.get('allergy3')
            prod.allergy4 = request.POST.get('allergy4')
            prod.allergy5 = request.POST.get('allergy5')
            prod.Q23 = request.POST.get('Q23')
            prod.Q24 = request.POST.get('Q24')
            #    item = questioner(Q0=Q0,Q1=Q1,Q2=Q2,Q3=Q3,Q4=Q4,Q5=Q5,Q6=Q6,Q7=Q7,Q8=Q8,Q9=Q9,Q10=Q10,Q11=Q11,
            #     Q12=Q12,Q13=Q13,Q14=Q14,Q15=Q15,Q16=Q16,Q17=Q17,Q18=Q18,Q19=Q19,Q20=Q20,Q21=Q21,Q22=Q22,allergy=allergy,
            #     Q23=Q23,Q24=Q24)
            prod.author = request.user
            prod.save()
            return redirect('/success-page/')
        else:
            return render(request, 'serverform.html', {'prod': prod})

    except questioner.DoesNotExist:
        prod = questioner()
        if request.method == 'POST':
            prod.Q0 = request.POST.get('Q0')
            prod.Q1 = request.POST.get('Q1')
            prod.Q2 = request.POST.get('Q2')
            prod.Q3 = request.POST.get('Q3')
            prod.Q4 = request.POST.get('Q4')
            prod.Q5 = request.POST.get('Q5')
            prod.Q6 = request.POST.get('Q6')
            prod.Q7 = request.POST.get('Q7')
            prod.Q8 = request.POST.get('Q8')
            prod.Q9 = request.POST.get('Q9')
            prod.Q10 = request.POST.get('Q10')
            prod.Q11 = request.POST.get('Q11')
            prod.Q12 = request.POST.get('Q12')
            prod.Q13 = request.POST.get('Q13')
            prod.Q14 = request.POST.get('Q14')
            prod.Q15 = request.POST.get('Q15')
            prod.Q16 = request.POST.get('Q16')
            prod.Q17 = request.POST.get('Q17')
            prod.Q18 = request.POST.get('Q18')
            prod.Q19 = request.POST.get('Q19')
            prod.Q20 = request.POST.get('Q20')
            prod.Q21 = request.POST.get('Q21')
            prod.Q22 = request.POST.get('Q22')
            prod.allergy = request.POST.get('allergy')
            prod.allergy1 = request.POST.get('allergy1')
            prod.allergy2 = request.POST.get('allergy2')
            prod.allergy3 = request.POST.get('allergy3')
            prod.allergy4 = request.POST.get('allergy4')
            prod.allergy5 = request.POST.get('allergy5')
            prod.Q23 = request.POST.get('Q23')
            prod.Q24 = request.POST.get('Q24')
            #    item = questioner(Q0=Q0,Q1=Q1,Q2=Q2,Q3=Q3,Q4=Q4,Q5=Q5,Q6=Q6,Q7=Q7,Q8=Q8,Q9=Q9,Q10=Q10,Q11=Q11,
            #     Q12=Q12,Q13=Q13,Q14=Q14,Q15=Q15,Q16=Q16,Q17=Q17,Q18=Q18,Q19=Q19,Q20=Q20,Q21=Q21,Q22=Q22,allergy=allergy,
            #     Q23=Q23,Q24=Q24)
            # instance = prod.save(commit=False)
            # instance.author = request.user
            # instance.save
            prod.author = request.user
            prod.save()
            return redirect('/success-page/')
        else:
            return render(request, 'serverform.html')


def success_page(request):
    try:
        prod = sideeffect.objects.get(author=request.user)
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.name = request.POST.get('name')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.name = request.POST.get('name')
            prod.author = request.user
            prod.save()
            return redirect('/side-effect-page/')
        else:
            return render(request, 'success.html', {'prod': prod})
    except sideeffect.DoesNotExist:
        prod = sideeffect()
        if request.method == 'POST'  and 'btnform1' in request.POST:
            prod.name = request.POST.get('name')
            prod.author = request.user
            prod.save()
            return redirect('/')
        if request.method == 'POST'  and 'btnform2' in request.POST:
            prod.name = request.POST.get('name')
            prod.author = request.user
            prod.save()
            return redirect('/side-effect-page/')
        else:
            return render(request, 'success.html')

def toggle_status(request, id):
    status = User.objects.get(id=id)
    status.is_active = 0
    status.save()
    return redirect('/dashboard/')


def toggle_status_active(request, id):
    status = User.objects.get(id=id)
    status.is_active = 1
    status.save()
    return redirect('/dashboard/')


# def dashboard(request):
#     item_list = user.objects.all().values().order_by('-date_created')
#     item_lists = sideeffect.objects.all().values()
#     quesT = questioner.objects.all().values()
#     userAccount = User.objects.all().values().filter(
#         is_superuser=False).order_by('-date_joined')
#     total_user = userAccount.count()
#     total_admin = User.objects.filter(is_superuser=True).count()

#     context = {
#         'item_list': item_list,
#         'item_lists': item_lists,
#         'quesT': quesT,
#         'userAccount': userAccount,
#         'total_user': total_user,
#         'total_admin': total_admin,
#     }

#     return render(request, 'system/dashboard.html', context)

#def dashboard(request):
#    #Total teen,adult and Senior
#    totalNoSymptoms = sideeffect.objects.filter(feverish = 0).values().count()
#    totalMild = sideeffect.objects.filter(feverish__gte=1 , feverish__lte=3).values().count()
#    totalModerate = sideeffect.objects.filter(feverish__gte=4 , feverish__lte=6).values().count()
#    totalSevere = sideeffect.objects.filter(feverish__gte=7).values().count()

#    childTotal = user.objects.filter(age__lte=17).values().count()
#    seniorTotal = user.objects.filter(age__gte=60).values().count()
#    adultTotal = user.objects.filter(age__gte=18 , age__lte=59).values().count()
#    myFilter = OrderFilter()
#    totalHaveSideeffect = sideeffect.objects.filter(name=1).count()
#    totalNoSideeffect = sideeffect.objects.filter(name=0).count()
#    item_list = user.objects.all().values().order_by('-date_created')
#    maleTotal = user.objects.filter(Q(gender='Male') | Q(gender='male')).values().count()
#    femaleTotal = user.objects.filter(Q(gender='Female') | Q(gender='female')).values().count()
#    item_lists = sideeffect.objects.all()
#    item_listsTotal = sideeffect.objects.all().values().count()
#    totalModerna = user.objects.filter(vaccination_brand='moderna').values().count()
#    totalPfizer = user.objects.filter(vaccination_brand='pfizer').values().count()
#    totalAstraZeneca = user.objects.filter(vaccination_brand='astraZeneca').values().count()
#    totalSinovac = user.objects.filter(vaccination_brand='sinovac').values().count()
#    totalJnj = user.objects.filter(vaccination_brand='johnson_and_Johnsons').values().count()
#    quesT = questioner.objects.all().values()
#    userAccount = User.objects.all().values().filter(is_superuser=False).order_by('-date_joined').order_by('-last_login')
#    total_user = user.objects.count()
#    total_admin = User.objects.filter(is_superuser=True).count()
#    chills = sideeffect.objects.filter(Q(chills='Yes') | Q(chills='yes') | Q(chills='1')).values().count()
#    fatigue = sideeffect.objects.filter(Q(fatigue='Yes') | Q(fatigue='yes') | Q(fatigue='1')).values().count()
#    feverTotal = sideeffect.objects.filter(Q(fever='Yes') | Q(fever='yes') | Q(fever='1')).values().count()
#    headache = sideeffect.objects.filter(Q(headache='Yes') | Q(headache='yes') | Q(headache='1')).values().count()
#    induration = sideeffect.objects.filter(Q(induration='Yes') | Q(induration='yes') | Q(induration='1')).values().count()
#    join_pain = sideeffect.objects.filter(Q(join_pain='Yes') | Q(join_pain='yes') | Q(join_pain='1')).values().count()
#    muscle_ache = sideeffect.objects.filter(Q(muscle_ache='Yes') | Q(muscle_ache='yes') | Q(muscle_ache='1')).values().count()
#    nausea = sideeffect.objects.filter(Q(nausea='Yes') | Q(nausea='yes') | Q(nausea='1')).values().count()
#    redness = sideeffect.objects.filter(Q(redness='Yes') | Q(redness='yes') | Q(redness='1')).values().count()
#    swelling = sideeffect.objects.filter(Q(swelling='Yes') | Q(swelling='yes') | Q(swelling='1')).values().count()
#    vomiting = sideeffect.objects.filter(Q(vomiting='Yes') | Q(vomiting='yes') | Q(vomiting='1')).values().count()
#    chillsN = sideeffect.objects.filter(Q(chills='No') | Q(chills='no') | Q(chills='0')).values().count()
#    fatigueN = sideeffect.objects.filter(Q(fatigue='No') | Q(fatigue='no') | Q(fatigue='0')).values().count()
#    feverTotalN = sideeffect.objects.filter(Q(fever='No') | Q(fever='no') | Q(fever='0')).values().count()
#    headacheN = sideeffect.objects.filter(Q(headache='No') | Q(headache='no') | Q(headache='0')).values().count()
#    indurationN = sideeffect.objects.filter(Q(induration='No') | Q(induration='no') | Q(induration='0')).values().count()
#    join_painN = sideeffect.objects.filter(Q(join_pain='No') | Q(join_pain='no') | Q(join_pain='0')).values().count()
#    muscle_acheN = sideeffect.objects.filter(Q(muscle_ache='No') | Q(muscle_ache='no') | Q(muscle_ache='0')).values().count()
#    nauseaN = sideeffect.objects.filter(Q(nausea='No') | Q(nausea='no') | Q(nausea='0')).values().count()
#    rednessN = sideeffect.objects.filter(Q(redness='No') | Q(redness='no') | Q(redness='0')).values().count()
#    swellingN = sideeffect.objects.filter(Q(swelling='No') | Q(swelling='no') | Q(swelling='0')).values().count()
#    vomitingN = sideeffect.objects.filter(Q(vomiting='No') | Q(vomiting='no') | Q(vomiting='0')).values().count()
#    num_days = 3
#    dates = [datetime.today().date() - timedelta(days=x) for x in range(num_days)]
#    # create a filter that includes multiple dates
#    date_filter = Q()
#    for date in dates:
#        date_filter |= Q(date_created=date)
#    dateTotal = user.objects.filter(date_filter).annotate(date=TruncDate('date_created')).values('date').annotate(total=Count('id')).order_by('-date')
#    dateTotals = dateTotal.count()
#    context = {
#        'totalNoSymptoms': totalNoSymptoms,
#        'totalMild': totalMild,
#        'totalModerate': totalModerate,
#        'totalSevere': totalSevere,
#        'totalHaveSideeffect': totalHaveSideeffect,
#        'totalNoSideeffect': totalNoSideeffect,
#        'dateTotals': dateTotals,
#        'dateTotal': dateTotal,
#        'childTotal': childTotal,
#        'adultTotal': adultTotal,
#        'seniorTotal': seniorTotal,
#        'item_list': item_list,
#        'item_lists': item_lists,
#        'item_listsTotal': item_listsTotal,
#        'quesT': quesT,
#        'userAccount': userAccount,
#        'total_user': total_user,
#        'total_admin': total_admin,
#        'maleTotal': maleTotal,
#        'femaleTotal': femaleTotal,
#        'totalModerna': totalModerna,
#        'totalPfizer': totalPfizer,
#        'totalAstraZeneca': totalAstraZeneca,
#        'totalSinovac': totalSinovac,
#        'totalJnj': totalJnj,
#        'chills': chills,
#        'fatigue': fatigue,
#        'feverTotal': feverTotal,
#        'headache': headache,
#        'induration': induration,
#        'join_pain': join_pain,
#        'muscle_ache': muscle_ache,
#        'nausea': nausea,
#        'redness': redness,
#        'swelling': swelling,
#        'vomiting': vomiting,
#        'chillsN':chillsN,
#        'fatigueN':fatigueN,
#        'feverTotalN':feverTotalN,
#        'headacheN':headacheN,
#        'indurationN':indurationN,
#        'join_painN':join_painN,
#        'muscle_acheN':muscle_acheN,
#        'nauseaN':nauseaN,
#        'rednessN':rednessN,
#        'swellingN':swellingN,
#        'vomitingN':vomitingN,
#    }
#    return render(request, 'home/index.html', context)

def dashboard(request):
    #Total teen,adult and Senior
    #totalNoSymptoms = sideeffect.objects.filter(feverish = 0).values().count()
    #totalMild = sideeffect.objects.filter(feverish__gte=1 , feverish__lte=3).values().count()
    #totalModerate = sideeffect.objects.filter(feverish__gte=4 , feverish__lte=6).values().count()
    #totalSevere = sideeffect.objects.filter(feverish__gte=7).values().count()
    #first dose
    countHeadacheMild = firstdose.objects.filter(headache = 'Mild').count()
    countInjectionSitePainMild = firstdose.objects.filter(InjectionSitePain = 'Mild').count()
    countfeverMild = firstdose.objects.filter(fever = 'Mild').count()
    countrashesMild = firstdose.objects.filter(rashes = 'Mild').count()
    countitchinessMild = firstdose.objects.filter(itchiness = 'Mild').count()
    countcoughMild = firstdose.objects.filter(cough = 'Mild').count()
    countbodyPainMild = firstdose.objects.filter(bodyPain = 'Mild').count()
    countsoarThroatMild = firstdose.objects.filter(soarThroat = 'Mild').count()
    countstomachAcheMild = firstdose.objects.filter(stomachAche = 'Mild').count()
    countvomitingMild = firstdose.objects.filter(vomiting = 'Mild').count()
    countdifficultyOfBreathingMild = firstdose.objects.filter(difficultyOfBreathing = 'Mild').count()
    countchestPainMild = firstdose.objects.filter(chestPain = 'Mild').count()
    countdisorentationMild = firstdose.objects.filter(disorentation = 'Mild').count()
    counttunnelVisionMild = firstdose.objects.filter(tunnelVision = 'Mild').count()
    countseizureMild = firstdose.objects.filter(seizure = 'Mild').count()
    #2nd dose
    countHeadacheMild2 = seconddose.objects.filter(headache = 'Mild').count()
    countInjectionSitePainMild2 = seconddose.objects.filter(InjectionSitePain = 'Mild').count()
    countfeverMild2 = seconddose.objects.filter(fever = 'Mild').count()
    countrashesMild2 = seconddose.objects.filter(rashes = 'Mild').count()
    countitchinessMild2 = seconddose.objects.filter(itchiness = 'Mild').count()
    countcoughMild2 = seconddose.objects.filter(cough = 'Mild').count()
    countbodyPainMild2 = seconddose.objects.filter(bodyPain = 'Mild').count()
    countsoarThroatMild2 = seconddose.objects.filter(soarThroat = 'Mild').count()
    countstomachAcheMild2 = seconddose.objects.filter(stomachAche = 'Mild').count()
    countvomitingMild2 = seconddose.objects.filter(vomiting = 'Mild').count()
    countdifficultyOfBreathingMild2 = seconddose.objects.filter(difficultyOfBreathing = 'Mild').count()
    countchestPainMild2 = seconddose.objects.filter(chestPain = 'Mild').count()
    countdisorentationMild2 = seconddose.objects.filter(disorentation = 'Mild').count()
    counttunnelVisionMild2 = seconddose.objects.filter(tunnelVision = 'Mild').count()
    countseizureMild2 = seconddose.objects.filter(seizure = 'Mild').count()
    #1st firstBooster
    countHeadacheMild3 = firstbooster.objects.filter(headache = 'Mild').count()
    countInjectionSitePainMild3 = firstbooster.objects.filter(InjectionSitePain = 'Mild').count()
    countfeverMild3 = firstbooster.objects.filter(fever = 'Mild').count()
    countrashesMild3 = firstbooster.objects.filter(rashes = 'Mild').count()
    countitchinessMild3 = firstbooster.objects.filter(itchiness = 'Mild').count()
    countcoughMild3 = firstbooster.objects.filter(cough = 'Mild').count()
    countbodyPainMild3 = firstbooster.objects.filter(bodyPain = 'Mild').count()
    countsoarThroatMild3 = firstbooster.objects.filter(soarThroat = 'Mild').count()
    countstomachAcheMild3 = firstbooster.objects.filter(stomachAche = 'Mild').count()
    countvomitingMild3 = firstbooster.objects.filter(vomiting = 'Mild').count()
    countdifficultyOfBreathingMild3 = firstbooster.objects.filter(difficultyOfBreathing = 'Mild').count()
    countchestPainMild3 = firstbooster.objects.filter(chestPain = 'Mild').count()
    countdisorentationMild3 = firstbooster.objects.filter(disorentation = 'Mild').count()
    counttunnelVisionMild3 = firstbooster.objects.filter(tunnelVision = 'Mild').count()
    countseizureMild3 = firstbooster.objects.filter(seizure = 'Mild').count()
    #2nd booster
    countHeadacheMild4 = secondbooster.objects.filter(headache = 'Mild').count()
    countInjectionSitePainMild4 = secondbooster.objects.filter(InjectionSitePain = 'Mild').count()
    countfeverMild4 = secondbooster.objects.filter(fever = 'Mild').count()
    countrashesMild4 = secondbooster.objects.filter(rashes = 'Mild').count()
    countitchinessMild4 = secondbooster.objects.filter(itchiness = 'Mild').count()
    countcoughMild4 = secondbooster.objects.filter(cough = 'Mild').count()
    countbodyPainMild4 = secondbooster.objects.filter(bodyPain = 'Mild').count()
    countsoarThroatMild4 = secondbooster.objects.filter(soarThroat = 'Mild').count()
    countstomachAcheMild4 = secondbooster.objects.filter(stomachAche = 'Mild').count()
    countvomitingMild4 = secondbooster.objects.filter(vomiting = 'Mild').count()
    countdifficultyOfBreathingMild4 = secondbooster.objects.filter(difficultyOfBreathing = 'Mild').count()
    countchestPainMild4 = secondbooster.objects.filter(chestPain = 'Mild').count()
    countdisorentationMild4 = secondbooster.objects.filter(disorentation = 'Mild').count()
    counttunnelVisionMild4 = secondbooster.objects.filter(tunnelVision = 'Mild').count()
    countseizureMild4 = secondbooster.objects.filter(seizure = 'Mild').count()

    #first dose
    countHeadacheSevere = firstdose.objects.filter(headache = 'Severe').count()
    countInjectionSitePainSevere = firstdose.objects.filter(InjectionSitePain = 'Severe').count()
    countfeverSevere = firstdose.objects.filter(fever = 'Severe').count()
    countrashesSevere = firstdose.objects.filter(rashes = 'Severe').count()
    countitchinessSevere = firstdose.objects.filter(itchiness = 'Severe').count()
    countcoughSevere = firstdose.objects.filter(cough = 'Severe').count()
    countbodyPainSevere = firstdose.objects.filter(bodyPain = 'Severe').count()
    countsoarThroatSevere = firstdose.objects.filter(soarThroat = 'Severe').count()
    countstomachAcheSevere = firstdose.objects.filter(stomachAche = 'Severe').count()
    countvomitingSevere = firstdose.objects.filter(vomiting = 'Severe').count()
    countdifficultyOfBreathingSevere = firstdose.objects.filter(difficultyOfBreathing = 'Severe').count()
    countchestPainSevere = firstdose.objects.filter(chestPain = 'Severe').count()
    countdisorentationSevere = firstdose.objects.filter(disorentation = 'Severe').count()
    counttunnelVisionSevere = firstdose.objects.filter(tunnelVision = 'Severe').count()
    countseizureSevere = firstdose.objects.filter(seizure = 'Severe').count()
    #2nd dose
    countHeadacheSevere2 = seconddose.objects.filter(headache = 'Severe').count()
    countInjectionSitePainSevere2 = seconddose.objects.filter(InjectionSitePain = 'Severe').count()
    countfeverSevere2 = seconddose.objects.filter(fever = 'Severe').count()
    countrashesSevere2 = seconddose.objects.filter(rashes = 'Severe').count()
    countitchinessSevere2 = seconddose.objects.filter(itchiness = 'Severe').count()
    countcoughSevere2 = seconddose.objects.filter(cough = 'Severe').count()
    countbodyPainSevere2 = seconddose.objects.filter(bodyPain = 'Severe').count()
    countsoarThroatSevere2 = seconddose.objects.filter(soarThroat = 'Severe').count()
    countstomachAcheSevere2 = seconddose.objects.filter(stomachAche = 'Severe').count()
    countvomitingSevere2 = seconddose.objects.filter(vomiting = 'Severe').count()
    countdifficultyOfBreathingSevere2 = seconddose.objects.filter(difficultyOfBreathing = 'Severe').count()
    countchestPainSevere2 = seconddose.objects.filter(chestPain = 'Severe').count()
    countdisorentationSevere2 = seconddose.objects.filter(disorentation = 'Severe').count()
    counttunnelVisionSevere2 = seconddose.objects.filter(tunnelVision = 'Severe').count()
    countseizureSevere2 = seconddose.objects.filter(seizure = 'Severe').count()
    #1st firstBooster
    countHeadacheSevere3 = firstbooster.objects.filter(headache = 'Severe').count()
    countInjectionSitePainSevere3 = firstbooster.objects.filter(InjectionSitePain = 'Severe').count()
    countfeverSevere3 = firstbooster.objects.filter(fever = 'Severe').count()
    countrashesSevere3 = firstbooster.objects.filter(rashes = 'Severe').count()
    countitchinessSevere3 = firstbooster.objects.filter(itchiness = 'Severe').count()
    countcoughSevere3 = firstbooster.objects.filter(cough = 'Severe').count()
    countbodyPainSevere3 = firstbooster.objects.filter(bodyPain = 'Severe').count()
    countsoarThroatSevere3 = firstbooster.objects.filter(soarThroat = 'Severe').count()
    countstomachAcheSevere3 = firstbooster.objects.filter(stomachAche = 'Severe').count()
    countvomitingSevere3 = firstbooster.objects.filter(vomiting = 'Severe').count()
    countdifficultyOfBreathingSevere3 = firstbooster.objects.filter(difficultyOfBreathing = 'Severe').count()
    countchestPainSevere3 = firstbooster.objects.filter(chestPain = 'Severe').count()
    countdisorentationSevere3 = firstbooster.objects.filter(disorentation = 'Severe').count()
    counttunnelVisionSevere3 = firstbooster.objects.filter(tunnelVision = 'Severe').count()
    countseizureSevere3 = firstbooster.objects.filter(seizure = 'Severe').count()
    #2nd booster
    countHeadacheSevere4 = secondbooster.objects.filter(headache = 'Severe').count()
    countInjectionSitePainSevere4 = secondbooster.objects.filter(InjectionSitePain = 'Severe').count()
    countfeverSevere4 = secondbooster.objects.filter(fever = 'Severe').count()
    countrashesSevere4 = secondbooster.objects.filter(rashes = 'Severe').count()
    countitchinessSevere4 = secondbooster.objects.filter(itchiness = 'Severe').count()
    countcoughSevere4 = secondbooster.objects.filter(cough = 'Severe').count()
    countbodyPainSevere4 = secondbooster.objects.filter(bodyPain = 'Severe').count()
    countsoarThroatSevere4 = secondbooster.objects.filter(soarThroat = 'Severe').count()
    countstomachAcheSevere4 = secondbooster.objects.filter(stomachAche = 'Severe').count()
    countvomitingSevere4 = secondbooster.objects.filter(vomiting = 'Severe').count()
    countdifficultyOfBreathingSevere4 = secondbooster.objects.filter(difficultyOfBreathing = 'Severe').count()
    countchestPainSevere4 = secondbooster.objects.filter(chestPain = 'Severe').count()
    countdisorentationSevere4 = secondbooster.objects.filter(disorentation = 'Severe').count()
    counttunnelVisionSevere4 = secondbooster.objects.filter(tunnelVision = 'Severe').count()
    countseizureSevere4 = secondbooster.objects.filter(seizure = 'Severe').count()
       
    totalMild = countHeadacheMild + countInjectionSitePainMild + countfeverMild + countrashesMild + countitchinessMild + countcoughMild + countbodyPainMild + countsoarThroatMild + countstomachAcheMild + countvomitingMild + countdifficultyOfBreathingMild + countchestPainMild + countdisorentationMild + counttunnelVisionMild + countseizureMild + countHeadacheMild2 + countInjectionSitePainMild2 + countfeverMild2 + countrashesMild2 + countitchinessMild2 + countcoughMild2 + countbodyPainMild2 + countsoarThroatMild2 + countstomachAcheMild2 + countvomitingMild2 + countdifficultyOfBreathingMild2 + countchestPainMild2 + countdisorentationMild2 + counttunnelVisionMild2 + countseizureMild2 + countHeadacheMild3 + countInjectionSitePainMild3 + countfeverMild3 + countrashesMild3 + countitchinessMild3 + countcoughMild3 + countbodyPainMild3 + countsoarThroatMild3 + countstomachAcheMild3 + countvomitingMild3 + countdifficultyOfBreathingMild3 + countchestPainMild3 + countdisorentationMild3 + counttunnelVisionMild3 + countseizureMild3 + countHeadacheMild4 + countInjectionSitePainMild4 + countfeverMild4 + countrashesMild4 + countitchinessMild4 + countcoughMild4 + countbodyPainMild4 + countsoarThroatMild4 + countstomachAcheMild4 + countvomitingMild4 + countdifficultyOfBreathingMild4 + countchestPainMild4 + countdisorentationMild4 + counttunnelVisionMild4 + countseizureMild4    
    totalSevere = countHeadacheSevere + countInjectionSitePainSevere + countfeverSevere + countrashesSevere + countitchinessSevere + countcoughSevere + countbodyPainSevere + countsoarThroatSevere + countstomachAcheSevere + countvomitingSevere + countdifficultyOfBreathingSevere + countchestPainSevere + countdisorentationSevere + counttunnelVisionSevere + countseizureSevere + countHeadacheSevere2 + countInjectionSitePainSevere2 + countfeverSevere2 + countrashesSevere2 + countitchinessSevere2 + countcoughSevere2 + countbodyPainSevere2 + countsoarThroatSevere2 + countstomachAcheSevere2 + countvomitingSevere2 + countdifficultyOfBreathingSevere2 + countchestPainSevere2 + countdisorentationSevere2 + counttunnelVisionSevere2 + countseizureSevere2 + countHeadacheSevere3 + countInjectionSitePainSevere3 + countfeverSevere3 + countrashesSevere3 + countitchinessSevere3 + countcoughSevere3 + countbodyPainSevere3 + countsoarThroatSevere3 + countstomachAcheSevere3 + countvomitingSevere3 + countdifficultyOfBreathingSevere3 + countchestPainSevere3 + countdisorentationSevere3 + counttunnelVisionSevere3 + countseizureSevere3 + countHeadacheSevere4 + countInjectionSitePainSevere4 + countfeverSevere4 + countrashesSevere4 + countitchinessSevere4 + countcoughSevere4 + countbodyPainSevere4 + countsoarThroatSevere4 + countstomachAcheSevere4 + countvomitingSevere4 + countdifficultyOfBreathingSevere4 + countchestPainSevere4 + countdisorentationSevere4 + counttunnelVisionSevere4 + countseizureSevere4    
    
    childTotal = user.objects.filter(age__lte=17).values().count()
    seniorTotal = user.objects.filter(age__gte=60).values().count()
    adultTotal = user.objects.filter(age__gte=18 , age__lte=59).values().count()
    myFilter = OrderFilter()
    totalHaveSideeffect = sideeffect.objects.filter(name=1).count()
    totalNoSideeffect = sideeffect.objects.filter(name=0).count()
    item_list = user.objects.all().values().order_by('-date_created')
    maleTotal = user.objects.filter(Q(gender='Male') | Q(gender='male')).values().count()
    femaleTotal = user.objects.filter(Q(gender='Female') | Q(gender='female')).values().count()
    item_lists = sideeffect.objects.all()
    item_listsTotal = sideeffect.objects.all().values().count()
    totalModerna = user.objects.filter(Q(vaccination_brand='moderna') | Q(vaccination_brand1='moderna') | Q(vaccination_brand2='moderna') | Q(vaccination_brand3='moderna')).values().count()
    totalPfizer = user.objects.filter(Q(vaccination_brand='pfizer') | Q(vaccination_brand1='pfizer') | Q(vaccination_brand2='pfizer') | Q(vaccination_brand3='pfizer')).values().count()
    totalAstraZeneca = user.objects.filter(Q(vaccination_brand='astraZeneca') | Q(vaccination_brand1='astraZeneca') | Q(vaccination_brand2='astraZeneca') | Q(vaccination_brand3='astraZeneca')).values().count()
    totalSinovac = user.objects.filter(Q(vaccination_brand='sinovac') | Q(vaccination_brand1='sinovac') | Q(vaccination_brand2='sinovac') | Q(vaccination_brand3='sinovac')).values().count()
    totalJnj = user.objects.filter(Q(vaccination_brand='johnson_and_Johnsons') | Q(vaccination_brand1='johnson_and_Johnsons') | Q(vaccination_brand2='johnson_and_Johnsons') | Q(vaccination_brand3='johnson_and_Johnsons')).values().count()
    userAccount = User.objects.all().values().filter(is_superuser=False).order_by('-date_joined').order_by('-last_login')
    total_user = user.objects.count()
    total_admin = User.objects.filter(is_superuser=True).count()
    chills = sideeffect.objects.filter(Q(chills='Yes') | Q(chills='yes') | Q(chills='1')).values().count()
    chillsN = sideeffect.objects.filter(Q(chills='No') | Q(chills='no') | Q(chills='0')).values().count()
    firstDoses = firstdose.objects.filter(user=1).count()
    secondDoses = seconddose.objects.filter(user=1).count()
    firstBoosters = firstbooster.objects.filter(user=1).count()
    secondBoosters = secondbooster.objects.filter(user=1).count()
    nfirstDoses = firstdose.objects.filter(user=0).count()
    nsecondDoses = seconddose.objects.filter(user=0).count()
    nfirstBoosters = firstbooster.objects.filter(user=0).count()
    nsecondBoosters = secondbooster.objects.filter(user=0).count()
    totalfirstDoses = user.objects.filter(firstDose=1).count()
    totalsecondDoses = user.objects.filter(secondDose=1).count()
    totalfirstBoosters = user.objects.filter(firstBooster=1).count()
    totalsecondBoosters = user.objects.filter(secondBooster=1).count()
    num_days = 3
    dates = [datetime.today().date() - timedelta(days=x) for x in range(num_days)]
    # create a filter that includes multiple dates
    date_filter = Q()
    for date in dates:
        date_filter |= Q(date_created=date)
    dateTotal = user.objects.filter(date_filter).annotate(date=TruncDate('date_created')).values('date').annotate(total=Count('id')).order_by('-date')
    dateTotalss = list(firstdose.objects.values_list('date', flat=True))
    dateTotals = dateTotal.count()
    context = {
        'dateTotalss': dateTotalss,
        'totalMild': totalMild,
         'nfirstDoses': nfirstDoses,
        'nsecondDoses': nsecondDoses,
        'nfirstBoosters': nfirstBoosters,
        'nsecondBoosters': nsecondBoosters,
        'totalfirstDoses': totalfirstDoses,
        'totalsecondDoses': totalsecondDoses,
        'totalfirstBoosters': totalfirstBoosters,
        'totalsecondBoosters': totalsecondBoosters,
        'firstBoosters': firstBoosters,
        'secondBoosters': secondBoosters,
        'firstDoses': firstDoses,
        'secondDoses': secondDoses,
        'totalMild': totalMild,
        'totalSevere': totalSevere,
        'totalHaveSideeffect': totalHaveSideeffect,
        'totalNoSideeffect': totalNoSideeffect,
        'dateTotals': dateTotals,
        'dateTotal': dateTotal,
        'childTotal': childTotal,
        'adultTotal': adultTotal,
        'seniorTotal': seniorTotal,
        'item_list': item_list,
        'item_lists': item_lists,
        'item_listsTotal': item_listsTotal,
        'userAccount': userAccount,
        'total_user': total_user,
        'total_admin': total_admin,
        'maleTotal': maleTotal,
        'femaleTotal': femaleTotal,
        'totalModerna': totalModerna,
        'totalPfizer': totalPfizer,
        'totalAstraZeneca': totalAstraZeneca,
        'totalSinovac': totalSinovac,
        'totalJnj': totalJnj,
        'chills': chills,
        'chillsN':chillsN,
    }
    return render(request, 'home/index.html', context)

def userAccount(request):
    userAccount = User.objects.all().values().filter(
        is_superuser=False).order_by('-date_joined')
    context = {
        'userAccount': userAccount,
    }
    return render(request, 'home/UserAccount.html', context)

    
def informationCollection(request):
    orders = user.objects.all()
    item_list = user.objects.all().values().order_by('-date_created')
    myFilter = OrderFilter(request.GET, queryset=orders)
    orders = myFilter.qs
    context = {
        'orders': orders,
        'myFilter': myFilter,
        'item_list': item_list,
    }

    return render(request, 'home/InformationCollection.html', context)


def survey(request):
    quesT = user.objects.all()
    context = {
        'quesT': quesT     
    }
    return render(request, 'home/vaccineInfo.html', context)

#def survey(request):
#    quesT = questioner.objects.all()
#    Q0 = quesT.filter(Q(Q0='Yes') | Q(Q0='yes') | Q(Q0='1')).count()
#    Q1 = quesT.filter(Q(Q1='Yes') | Q(Q1='yes') | Q(Q1='1')).count()
#    Q2 = quesT.filter(Q(Q2='Yes') | Q(Q2='yes') | Q(Q2='1')).count()
#    Q3 = quesT.filter(Q(Q3='Yes') | Q(Q3='yes') | Q(Q3='1')).count()
#    Q4 = quesT.filter(Q(Q4='Yes') | Q(Q4='yes') | Q(Q4='1')).count()
#    Q5 = quesT.filter(Q(Q5='Yes') | Q(Q5='yes') | Q(Q5='1')).count()
#    Q6 = quesT.filter(Q(Q6='Yes') | Q(Q6='yes') | Q(Q6='1')).count()
#    Q7 = quesT.filter(Q(Q7='Yes') | Q(Q7='yes') | Q(Q7='1')).count()
#    Q8 = quesT.filter(Q(Q8='Yes') | Q(Q8='yes') | Q(Q8='1')).count()
#    Q9 = quesT.filter(Q(Q9='Yes') | Q(Q9='yes') | Q(Q9='1')).count()
#    Q10 = quesT.filter(Q(Q10='Yes') | Q(Q10='yes') | Q(Q10='1')).count()
#    Q11 = quesT.filter(Q(Q11='Yes') | Q(Q11='yes') | Q(Q11='1')).count()
#    Q12 = quesT.filter(Q(Q12='Yes') | Q(Q12='yes') | Q(Q12='1')).count()
#    Q13 = quesT.filter(Q(Q13='Yes') | Q(Q13='yes') | Q(Q13='1')).count()
#    Q14 = quesT.filter(Q(Q14='Yes') | Q(Q14='yes') | Q(Q14='1')).count()
#    Q15 = quesT.filter(Q(Q15='Yes') | Q(Q15='yes') | Q(Q15='1')).count()
#    Q16 = quesT.filter(Q(Q16='Yes') | Q(Q16='yes') | Q(Q16='1')).count()
#    Q17 = quesT.filter(Q(Q17='Yes') | Q(Q17='yes') | Q(Q17='1')).count()
#    Q18 = quesT.filter(Q(Q18='Yes') | Q(Q18='yes') | Q(Q18='1')).count()
#    Q19 = quesT.filter(Q(Q19='Yes') | Q(Q19='yes') | Q(Q19='1')).count()
#    Q20 = quesT.filter(Q(Q20='Yes') | Q(Q20='yes') | Q(Q20='1')).count()
#    Q21 = quesT.filter(Q(Q21='Yes') | Q(Q21='yes') | Q(Q21='1')).count()
#    Q22 = quesT.filter(Q(Q22='Yes') | Q(Q22='yes') | Q(Q22='1')).count()
#    Q23 = quesT.filter(Q(Q23='Yes') | Q(Q23='yes') | Q(Q23='1')).count()
#    Q24 = quesT.filter(Q(Q24='Yes') | Q(Q24='yes') | Q(Q24='1')).count()
#    allergy2 = quesT.filter(Q(allergy2='Latex Allergy')).count()
#    allergy3 = quesT.filter(Q(allergy3='Mold Allergy')).count()
#    allergy4 = quesT.filter(Q(allergy4='Pet Allergy')).count()
#    allergy5 = quesT.filter(Q(allergy5='Pollen Allergy')).count()
#    context = {
#        'quesT': quesT,
#        'allergyss': allergy2,
#        'allergysss': allergy3,
#        'allergyssss': allergy4,
#        'allergysssss': allergy5,
#        'Q0': Q0,
#        'Q1': Q1,
#        'Q2': Q2,
#        'Q3': Q3,
#        'Q4': Q4,
#        'Q5': Q5,
#        'Q6': Q6,
#        'Q7': Q7,
#        'Q8': Q8,
#        'Q9': Q9,
#        'Q10': Q10,
#        'Q11': Q11,
#        'Q12': Q12,
#        'Q13': Q13,
#        'Q14': Q14,
#        'Q15': Q15,
#        'Q16': Q16,
#        'Q17': Q17,
#        'Q18': Q18,
#        'Q19': Q19,
#        'Q20': Q20,
#        'Q21': Q21,
#        'Q22': Q22,
#        'Q23': Q23,
#        'Q24': Q24,
        
#    }
#    return render(request, 'home/Survey.html', context)
#def count_symptoms_by_row(side_effects):
#    symptom_counts = []
#    for effect in side_effects:
#        count = 0
#        if effect.muscle_ache == 'Yes':
#            count += 1
#        if effect.headache == 'Yes':
#            count += 1
#        if effect.fever == 'Yes':
#            count += 1
#        if effect.redness == 'Yes':
#            count += 1
#        if effect.swelling == 'Yes':
#            count += 1
#        if effect.tenderness == 'Yes':
#            count += 1
        
      
#    return symptom_counts

def sideEffectReports(request):
    haveSideeffect = user.objects.all().values()

    s = sideeffect.objects.all()
    myFilters = EffectFilter(request.GET, queryset=haveSideeffect)
    orders = myFilters.qs
    context = {
    's': s,
    'orders': orders,
    'myFilters': myFilters,
    'haveSideeffect': haveSideeffect,
    } 
    print(s);
    return render(request, 'home/SideEffectReports.html', context)

def sideEffect(request):
    firstdosage = firstdose.objects.all()
    seconddosage = seconddose.objects.all()
    firstboosters = firstbooster.objects.all()
    secondboosters = secondbooster.objects.all()

    context = {
        'firstdosage': firstdosage,
        'seconddosage': seconddosage,
        'firstboosters': firstboosters,
        'secondboosters': secondboosters
    }
    return render(request, 'home/SideEffect.html', context)

def sideeffectFirstDose(request):
    firstdosage = firstdose.objects.filter(user=1)
    firstdosageOut = firstdose.objects.filter(user=0)
    context = {
        'firstdosage': firstdosage,
        'firstdosageOut': firstdosageOut,
    }
    return render(request, 'home/firstDose.html', context)
def sideEffectSecondDose(request):
    seconddosage = seconddose.objects.filter(user=1)
    seconddosageOut = seconddose.objects.filter(user=0)
    context = {
        'seconddosage': seconddosage,
        'seconddosageOut': seconddosageOut,
    }
    return render(request, 'home/secondDose.html', context)
def sideEffectFirstBooster(request):
    firstboosters = firstbooster.objects.filter(user=1)
    firstboostersOut = firstbooster.objects.filter(user=0)
    context = {
        'firstboosters': firstboosters,
        'firstboostersOut': firstboostersOut,
    }
    return render(request, 'home/firstBooster.html', context)
def sideEffectSecondBooster(request):
    secondboosters = secondbooster.objects.filter(user=1)
    secondboostersOut = secondbooster.objects.filter(user=0)
    context = {
        'secondboosters': secondboosters,
        'secondboostersOut': secondboostersOut,
    }
    return render(request, 'home/secondBooster.html', context)
# def dashboard(request):
#     item_list = user.objects.all().values().order_by('-date_created')
#     item_lists = sideeffect.objects.all().values()
#     quesT = questioner.objects.all().values()
#     userAccount = User.objects.all().values().filter(
#         is_superuser=False).order_by('-date_joined')
#     total_user = userAccount.count()
#     total_admin = User.objects.filter(is_superuser=True).count()

#     context = {
#         'item_list': item_list,
#         'item_lists': item_lists,
#         'quesT': quesT,
#         'userAccount': userAccount,
#         'total_user': total_user,
#         'total_admin': total_admin,
#     }

#     return render(request, 'layouts/LayoutDashboard.html', context)

    # template = loader.get_template('system/dashboard.html')
    # return HttpResponse(template.render())

# def delete(request):
#     prod = User.objects.get(pk=request.user.id)
#     prod.is_active = True
#     prod.save()
#     return render(request,'system/dashboard.html')
# def delete(request):
#    prod = questioner.objects.get(pk=request.user.id)
#    prod.delete()
#    return redirect (request, 'dashboard/')
# def edit(request,id):
#     users = user.objects.get(pk=id)
#     form = userForm(request.POST or None,instance = users)
#     return render(request,'information.html',{'form':form })
