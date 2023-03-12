<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.24/css/jquery.dataTables.min.css">

<table id="projects" class="display" style="width:100%"></table>

<script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
<script>
$(document).ready(function() {
    $('#projects').DataTable({
        ajax: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vS02TfroKkbnydEBUjwO8gEYWgBfuAjWXCpSp_YVMvNV_YSDq-elWdyPdsLV8-6b-u9LJMgchhfnsQ-/pub?gid=0&single=true&output=csv',
        columns: [
            {data: '#'},
            {data: 'ID'},
            {data: 'URL'},
            {data: 'Dataset'},
            {data: 'Data source'},
            {data: 'Problem type'},
            {data: 'Time series data'},
            {data: 'Image data'},
            {data: 'Video'},
            {data: 'Additional info'}
        ],
    });
});
</script>
