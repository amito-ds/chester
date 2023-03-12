<table id="projects" class="display" style="width:100%">
</table>

<script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
<script>
$(document).ready(function() {
    $('#my-table').DataTable({
        ajax: '/Users/amitosi/PycharmProjects/chester/projects/project_example.csv',
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
