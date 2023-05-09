$('.btnNext').click(function () {
    const nextTabLinkEl = $('.nav-tabs .active').closest('li').next('li').find('a')[0];
    const nextTab = new bootstrap.Tab(nextTabLinkEl);
    nextTab.show();
});

$('.btnPrevious').click(function () {
    const prevTabLinkEl = $('.nav-tabs .active').closest('li').prev('li').find('a')[0];
    const prevTab = new bootstrap.Tab(prevTabLinkEl);
    prevTab.show();
});

function handleValueChange() {
    var y = document.getElementById('name').value;
    var x = document.getElementById('nameResult');
    x.innerHTML = y;
}
function handleValueChange1() {
    var y = document.getElementById('vaccination_brand').value;
    var x = document.getElementById('brandResult');
    x.innerHTML = y;
}
function handleValueChange2() {
    var y = document.getElementById('vaccination_site').value;
    var x = document.getElementById('siteResult');
    x.innerHTML = y;
}
function handleValueChange3() {
    var y = document.getElementById('address').value;
    var x = document.getElementById('addressResult');
    x.innerHTML = y;
}
function handleValueChange4() {
    var y = document.getElementById('contact_number').value;
    var x = document.getElementById('contactResult');
    x.innerHTML = y;
}
function handleValueChange5() {
    var y = document.getElementById('bday').value;
    var x = document.getElementById('bdayResult');
    x.innerHTML = y;
}
function handleValueChange6() {
    var y = document.getElementById('age').value;
    var x = document.getElementById('ageResult');
    x.innerHTML = y;
}
function handleValueChange7() {
    var y = document.getElementById('gender').value;
    var x = document.getElementById('genderResult');
    x.innerHTML = y;
}

//side effect
function sideEffect1() {
    var y = document.getElementById('txt_check_1_input1').value;
    var x = document.getElementById('Muscle');
    x.innerHTML = y;
}
function sideEffect2() {
    var y = document.getElementById('txt_check_2_input1').value;
    var x = document.getElementById('Headache');
    x.innerHTML = y;
}
function sideEffect3() {
    var y = document.getElementById('txt_check_3_input1').value;
    var x = document.getElementById('Fever');
    x.innerHTML = y;
}
function sideEffect4() {
    var y = document.getElementById('txt_check_4_input1').value;
    var x = document.getElementById('Redness');
    x.innerHTML = y;
}
function sideEffect5() {
    var y = document.getElementById('txt_check_5_input1').value;
    var x = document.getElementById('Swelling');
    x.innerHTML = y;
}
function sideEffect6() {
    var y = document.getElementById('txt_check_6_input1').value;
    var x = document.getElementById('Tenderness');
    x.innerHTML = y;
}
function sideEffect7() {
    var y = document.getElementById('txt_check_7_input1').value;
    var x = document.getElementById('Warmth');
    x.innerHTML = y;
}
function sideEffect8() {
    var y = document.getElementById('txt_check_8_input1').value;
    var x = document.getElementById('Itch');
    x.innerHTML = y;
}
function sideEffect9() {
    var y = document.getElementById('txt_check_9_input1').value;
    var x = document.getElementById('Induration');
    x.innerHTML = y;
}
function sideEffect10() {
    var y = document.getElementById('txt_check_10_input1').value;
    var x = document.getElementById('Feverish');
    x.innerHTML = y;
}
function sideEffect11() {
    var y = document.getElementById('txt_check_11_input1').value;
    var x = document.getElementById('Chills');
    x.innerHTML = y;
}
function sideEffect12() {
    var y = document.getElementById('txt_check_12_input1').value;
    var x = document.getElementById('Joint');
    x.innerHTML = y;
}
function sideEffect13() {
    var y = document.getElementById('txt_check_13_input1').value;
    var x = document.getElementById('Fatigue');
    x.innerHTML = y;
}
function sideEffect14() {
    var y = document.getElementById('txt_check_14_input1').value;
    var x = document.getElementById('Nausea');
    x.innerHTML = y;
} function sideEffect15() {
    var y = document.getElementById('txt_check_15_input1').value;
    var x = document.getElementById('Vomiting');
    x.innerHTML = y;
}

//survey
function survey1() {
    var y = document.getElementById('Q0').value;
    var x = document.getElementById('Q0result');
    x.innerHTML = y;
}
function survey2() {
    var y = document.getElementById('Q1').value;
    var x = document.getElementById('Q1result');
    x.innerHTML = y;
}
function survey3() {
    var y = document.getElementById('Q2').value;
    var x = document.getElementById('Q2result');
    x.innerHTML = y;
}
function survey4() {
    var y = document.getElementById('Q3').value;
    var x = document.getElementById('Q3result');
    x.innerHTML = y;
}
function survey5() {
    var y = document.getElementById('Q4').value;
    var x = document.getElementById('Q4result');
    x.innerHTML = y;
}
function survey6() {
    var y = document.getElementById('Q5').value;
    var x = document.getElementById('Q5result');
    x.innerHTML = y;
}
function survey7() {
    var y = document.getElementById('Q6').value;
    var x = document.getElementById('Q6result');
    x.innerHTML = y;
}
function survey8() {
    var y = document.getElementById('Q7').value;
    var x = document.getElementById('Q7result');
    x.innerHTML = y;
}
function survey9() {
    var y = document.getElementById('Q8').value;
    var x = document.getElementById('Q8result');
    x.innerHTML = y;
}
function survey10() {
    var y = document.getElementById('Q9').value;
    var x = document.getElementById('Q9result');
    x.innerHTML = y;
}
function survey11() {
    var y = document.getElementById('Q10').value;
    var x = document.getElementById('Q10result');
    x.innerHTML = y;
}
function survey12() {
    var y = document.getElementById('Q11').value;
    var x = document.getElementById('Q11result');
    x.innerHTML = y;
}
function survey13() {
    var y = document.getElementById('Q12').value;
    var x = document.getElementById('Q12result');
    x.innerHTML = y;
}
function survey14() {
    var y = document.getElementById('Q13').value;
    var x = document.getElementById('Q13result');
    x.innerHTML = y;
}
function survey15() {
    var y = document.getElementById('Q14').value;
    var x = document.getElementById('Q14result');
    x.innerHTML = y;
}
function survey16() {
    var y = document.getElementById('Q15').value;
    var x = document.getElementById('Q15result');
    x.innerHTML = y;
}
function survey17() {
    var y = document.getElementById('Q16').value;
    var x = document.getElementById('Q16result');
    x.innerHTML = y;
}
function survey18() {
    var y = document.getElementById('Q17').value;
    var x = document.getElementById('Q17result');
    x.innerHTML = y;
}
function survey19() {
    var y = document.getElementById('Q18').value;
    var x = document.getElementById('Q18result');
    x.innerHTML = y;
}
function survey20() {
    var y = document.getElementById('Q19').value;
    var x = document.getElementById('Q19result');
    x.innerHTML = y;
}
function survey21() {
    var y = document.getElementById('Q20').value;
    var x = document.getElementById('Q20result');
    x.innerHTML = y;
}
function survey22() {
    var y = document.getElementById('Q21').value;
    var x = document.getElementById('Q21result');
    x.innerHTML = y;
}
function survey23() {
    var y = document.getElementById('Q22').value;
    var x = document.getElementById('Q22result');
    x.innerHTML = y;
}
function survey24() {
    var y = document.getElementById('Q23').value;
    var x = document.getElementById('Q23result');
    x.innerHTML = y;
}
function survey25() {
    var y = document.getElementById('Q24').value;
    var x = document.getElementById('Q24result');
    x.innerHTML = y;
}
// allergy
function allergy() {
    var y = document.getElementById('allergy').value;
    var x = document.getElementById('allergyResult');
    x.innerHTML = y;
}
function allergy1() {
    var y = document.getElementById('allergy1').value;
    var x = document.getElementById('allergy1Result');
    x.innerHTML = y;
}
function allergy2() {
    var y = document.getElementById('allergy2').value;
    var x = document.getElementById('allergy2Result');
    x.innerHTML = y;
}
function allergy3() {
    var y = document.getElementById('allergy3').value;
    var x = document.getElementById('allergy3Result');
    x.innerHTML = y;
}
function allergy4() {
    var y = document.getElementById('allergy4').value;
    var x = document.getElementById('allergy4Result');
    x.innerHTML = y;
}
function allergy5() {
    var y = document.getElementById('allergy5').value;
    var x = document.getElementById('allergy5Result');
    x.innerHTML = y;
}
