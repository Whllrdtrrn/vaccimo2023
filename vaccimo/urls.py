from django.urls import path
from django.contrib import admin
from vaccimo import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('reset_password/', auth_views.PasswordResetView.as_view(template_name='passwordReset/change-password.html'), name = 'passwordReset'),
    path('reset_password_sent/', auth_views.PasswordResetDoneView.as_view(template_name='passwordReset/password_reset_done.html'), name = 'password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='passwordReset/confirm.html'), name = 'password_reset_confirm'),
    path('reset_password_complete/', auth_views.PasswordResetCompleteView.as_view(template_name='passwordReset/complete.html'), name = 'password_reset_complete'),

    # pdf 
    path('pdf/<int:pk>/', views.generate_pdf, name='generate_pdf'),
    path('pdf/', views.pdfAllReport, name='pdfAllReport'),

    #     path('admin/', admin.site.urls),
    path('', views.LayoutHome, name='LayoutHome'),
    #     path('admin/', views.homepage, name='homepage'),
    #path('vaccimo/', views.LayoutIndex, name='LayoutIndex'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/user-account/', views.userAccount, name='userAccount'),
    path('dashboard/information-collection/',
         views.informationCollection, name='informationCollection'),
    path('dashboard/survey/', views.survey, name='survey'),
    path('dashboard/side-effect/', views.sideEffect, name='sideEffect'),
    path('dashboard/side-effect/first-dose', views.sideeffectFirstDose, name='sideeffectFirstDose'),
    path('dashboard/side-effect/second-dose', views.sideEffectSecondDose, name='sideEffectSecondDose'),
    path('dashboard/side-effect/first-booster', views.sideEffectFirstBooster, name='sideEffectFirstBooster'),
    path('dashboard/side-effect/second-booster', views.sideEffectSecondBooster, name='sideEffectSecondBooster'),

    path('dashboard/side-effect-reports/', views.sideEffectReports, name='sideEffectReports'),
    
    path('reports/', views.reports, name='reports'),

    #path('vaccimo/follow-up/', views.followUp, name='followUp'),
    path('vaccimo/', views.informationNew, name='informationNew'),
    path('vaccimo/firstDose/', views.firstDose, name='firstDose'),
    path('vaccimo/secondDoseQ/', views.secondDoseQ, name='secondDoseQ'),
    path('vaccimo/secondDose/', views.secondDose, name='secondDose'),
    path('vaccimo/survey/', views.surveyNew, name='surveyNew'),
    path('vaccimo/whatDoYouFeel/', views.whatDoYouFeel, name='whatDoYouFeel'),
       path('vaccimo/whatDoYouFeel2/', views.whatDoYouFeel2, name='whatDoYouFeel'),
       path('vaccimo/whatDoYouFeel3/', views.whatDoYouFeel3, name='whatDoYouFeel'),
       path('vaccimo/whatDoYouFeel4/', views.whatDoYouFeel4, name='whatDoYouFeel'),
    path('vaccimo/side-effect-new/', views.sideEffectNew, name='sideEffectNew'),
    path('vaccimo/side-effect-new2/', views.sideEffectNew2, name='sideEffectNew2'),
    path('vaccimo/side-effect-new3/', views.sideEffectNew3, name='sideEffectNew3'),
    path('vaccimo/side-effect-new4/', views.sideEffectNew4, name='sideEffectNew4'),
    path('vaccimo/firstBoosterQuestion/', views.firstBoosterQuestion, name='firstBoosterQuestion'),
    path('vaccimo/firstBooster/', views.firstBooster, name='firstBooster'),
    path('vaccimo/secondBoosterQuestion/', views.secondBoosterQuestion, name='secondBoosterQuestion'),
    path('vaccimo/secondBooster/', views.secondBooster, name='secondBooster'),

    path('', views.home_page, name='homepage'),
    path('services/', views.services, name='services'),
    path('contacts/', views.contacts, name='contacts'),
    path('aboutUs/', views.aboutUs, name='aboutUs'),
    path('loginpage/', views.login_page, name='loginpage'),
    path('register/', views.register_page, name='register'),
    path('view-details/', views.viewDetails, name='viewDetails'),
    path('information-page/', views.information_page, name='information'),
    path('Profile/', views.profileEdit, name='profileEdit'),
    path('server-form-page/', views.server_form, name='server'),
    path('side-effect-page/', views.sideeffect_page, name='sideeffect'),
    path('success-page/', views.success_page, name='success'),
    #     path('dashboard/', views.dashboard, name='dashboard'),
    path('toggle_status/<str:id>', views.toggle_status, name='toggle_status'),
    path('toggle_status_active/<str:id>',
         views.toggle_status_active, name='toggle_status_active'),
    path('logout/', views.logout_view, name='logout'),
    path('logout1/', views.logout_view1, name='logout1'),
    path('logout2/', views.logout_view2, name='logout2'),
    path('preprocessing/', views.preprocessing, name='preprocessing'),
    path('checker_page/', views.checker_page, name='checker_page'),
    path('chooseMethod/', views.chooseMethod, name='chooseMethod'),
    # path('classification/', views.classification, name='classification'),
    path('clustering/', views.clustering, name='clustering'),
    path('activate/<uidb64>/<token>/',views.activate, name='activate'),  
    path('email-verification/', views.verification, name='verification'),

    # path('delete/', views.delete, name='delete'),
    # path('edit/<str:pk>', views.edit, name='edit') ,

]
