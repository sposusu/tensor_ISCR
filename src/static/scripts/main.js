
function get_big5() {
	var q = $('#query').val();
	console.log("Query value:" + q)

	var data = {
		"action": "firstpass",
		"query": q
	}

		
	$.post('/query', data, function(data){
		$(document).find('tbody').empty();
		$.each(data, function(i, item) {
			console.log(i,item);
			$('#result').find('tbody')
				.append($('<tr>')
					.append($('<td>').append(i),
						$('<td>').append(item.big5),
						$('<td>').append(item.probability)
					)
				)
		})
	})
	$('#query').val('');
	return false
}

$(document).ready( function() {
	$('#query').keypress(function(e){
		console.log('key pressed!');
		if(e.which == 13 && !e.shiftKey){
			console.log('get_big5');
			e.preventDefault();
			get_big5();
		}
	})

	console.log('ready!');
})
