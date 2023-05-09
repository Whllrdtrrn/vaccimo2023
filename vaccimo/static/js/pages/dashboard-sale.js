'use strict';
document.addEventListener("DOMContentLoaded", function () {
    setTimeout(function () {
        floatchart()
    }, 700);
    // [ campaign-scroll ] start
    var px = new PerfectScrollbar('.feed-scroll', {
        wheelSpeed: .5,
        swipeEasing: 0,
        wheelPropagation: 1,
        minScrollbarLength: 40,
    });
    var px = new PerfectScrollbar('.pro-scroll', {
        wheelSpeed: .5,
        swipeEasing: 0,
        wheelPropagation: 1,
        minScrollbarLength: 40,
    });
    // [ campaign-scroll ] end
});

function floatchart() {
    // [ support-chart ] start
    (function () {
        var options1 = {
            chart: {
                type: 'area',
                height: 85,
                sparkline: {
                    enabled: true
                }
            },
            colors: ["#7267EF"],
            stroke: {
                curve: 'smooth',
                width: 2,
            },
            series: [{
                data: [0, 20, 10, 45, 30, 55, 20, 30, 0]
            }],
            tooltip: {
                fixed: {
                    enabled: false
                },
                x: {
                    show: false
                },
                y: {
                    title: {
                        formatter: function (seriesName) {
                            return 'Ticket '
                        }
                    }
                },
                marker: {
                    show: false
                }
            }
        }
        new ApexCharts(document.querySelector("#support-chart"), options1).render();
        var options2 = {
            chart: {
                type: 'bar',
                height: 85,
                sparkline: {
                    enabled: true
                }
            },
            colors: ["#7267EF"],
            plotOptions: {
                bar: {
                    columnWidth: '70%'
                }
            },
            series: [{
                data: [25, 66, 41, 89, 63, 25, 44, 12, 36, 9, 54, 44, 12, 36, 9, 54, 25, 66, 41, 89, 63, 25, 44, 12, 36, 9, 25, 44, 12, 36, 9, 54]
            }],
            xaxis: {
                crosshairs: {
                    width: 1
                },
            },
            tooltip: {
                fixed: {
                    enabled: false
                },
                x: {
                    show: false
                },
                y: {
                    title: {
                        formatter: function (seriesName) {
                            return ''
                        }
                    }
                },
                marker: {
                    show: false
                }
            }
        }
        new ApexCharts(document.querySelector("#support-chart1"), options2).render();
    })();
    // [ support-chart ] end
    // [ account-chart ] start

    // (function () {
    //     var options = {
    //         chart: {
    //             height: 350,
    //             type: 'line',
    //             stacked: false,
    //         },
    //         stroke: {
    //             width: [0, 3],
    //             curve: 'smooth'
    //         },
    //         plotOptions: {
    //             bar: {
    //                 columnWidth: '20%'
    //             }
    //         },
    //         colors: ['#7267EF', '#c7d9ff'],
    //         series: [{
    //             name: 'Total',
    //             type: 'column',
    //             data: [totalModerna, 25, 36, 30, 45, 35, 64, 52, 59, 36, 39, 12, 21, 32, 2]
    //         }, {
    //             name: 'Average',
    //             type: 'line',
    //             data: [30, 25, 36, 30, 45, 35, 64, 52, 59, 36, 39, 12, 21, 32, 2]
    //         }],
    //         fill: {
    //             opacity: [0.85, 1],
    //         },
    //         labels: ['Muscle Ache', 'Fatigue', 'Fever', 'Feverish', 'Headache', 'Induration', 'Itch', 'Join Pain', 'Nausea', 'Redness', 'Swelling', 'Tenderness', 'Vomiting', 'Warmth'],
    //         markers: {
    //             size: 0
    //         },
    //         xaxis: {
    //             type: 'text'
    //         },
    //         yaxis: {
    //             min: 0
    //         },
    //         tooltip: {
    //             shared: true,
    //             intersect: false,
    //             y: {
    //                 formatter: function (y) {
    //                     if (typeof y !== "undefined") {
    //                         return "$ " + y.toFixed(0);
    //                     }
    //                     return y;

    //                 }
    //             }
    //         },
    //         legend: {
    //             labels: {
    //                 useSeriesColors: true
    //             },
    //             markers: {
    //                 customHTML: [
    //                     function () {
    //                         return ''
    //                     },
    //                     function () {
    //                         return ''
    //                     }
    //                 ]
    //             }
    //         }
    //     }
    //     var chart = new ApexCharts(
    //         document.querySelector("#account-chart"),
    //         options
    //     );
    //     chart.render();
    // })();

    // [ account-chart ] end
    // [ satisfaction-chart ] start
    //(function () {
    //    var options = {
    //        chart: {
    //            height: 260,
    //            type: 'pie',
    //        },
    //        series: ['totalModerna', 'totalModerna', 'totalModerna', 'totalModerna', 'totalModerna'],
    //        labels: ["Moderna", "Pfizer", "AstraZeneca", "Sinovac", "Johnson and Johnson's"],
    //        legend: {
    //            show: true,
    //            offsetY: 50,
    //        },
    //        dataLabels: {
    //            enabled: true,
    //            dropShadow: {
    //                enabled: false,
    //            }
    //        },
    //        theme: {
    //            monochrome: {
    //                enabled: true,
    //                color: '#7267EF',
    //            }
    //        },
    //        responsive: [{
    //            breakpoint: 768,
    //            options: {
    //                chart: {
    //                    height: 320,

    //                },
    //                legend: {
    //                    position: 'bottom',
    //                    offsetY: 0,
    //                }
    //            }
    //        }]
    //    }
    //    var chart = new ApexCharts(document.querySelector("#satisfaction-chart"), options);
    //    chart.render();
    //})();
    // [ satisfaction-chart ] end
}