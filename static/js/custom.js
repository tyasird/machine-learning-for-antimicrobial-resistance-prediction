jQuery(document).ready(function ($) {

    "use strict";

    $('.preloader').hide()


    $("#do_analyze").click(function () {
        var $myForm = $('#analyze_form');
        if ($myForm[0].checkValidity()) {
            $('.preloader').show()
            $myForm.find(':submit').click();
        }
    });


    $('#select_all').click(function () {
        $('#labels option').prop('selected', true);
    });


    $('#deselect_all').click(function () {
        $('#labels option').prop('selected', false);
    });


});