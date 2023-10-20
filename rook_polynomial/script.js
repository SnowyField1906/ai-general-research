/**
*
* @Name : RookPolynomial/script.js
* @Version : 1.0
* @Programmer : Max
* @Date : 2019-05-26
* @Released under : https://github.com/BaseMax/RookPolynomial/blob/master/LICENSE
* @Repository : https://github.com/BaseMax/RookPolynomial
*
**/
let chess=[]
let sections = document.querySelectorAll(".section")
let sectionForm = document.querySelector(".section-form")
let sectionBoard = document.querySelector(".section-board")
let sectionRook = document.querySelector(".section-rook")
let inputSizeWidthValue
let inputSizeHeightValue
let chessSize
let inputSizeWidth = sectionForm.querySelector(".input-size-width")
let inputSizeHeight = sectionForm.querySelector(".input-size-height") 
let inputSizeButton = sectionForm.querySelector(".input-size-button")
let boards=document.querySelectorAll(".board")
inputSizeButton.addEventListener("click", function() {
	inputSizeWidthValue = inputSizeWidth.value
	inputSizeHeightValue = inputSizeHeight.value
	chessSize = inputSizeWidthValue * inputSizeHeightValue
	let chessTable = ""
	// for(let i=0 i<chessSize i++) {
	// 	let index=i+1
	// 	let row
	// 	let column
	// 	if(index % inputSizeWidthValue === 0) {
	// 		row=Math.floor(index / inputSizeWidthValue)
	// 	}
	// 	else {
	// 		row=Math.floor(index / inputSizeWidthValue) + 1
	// 	}
	// 	column=index - ((row-1) * inputSizeHeightValue)
	// 	if(column === 0) {
	// 		column++
	// 	}
	// 	chessTable+='<div class="column" data-row="'+ row +'" data-column="'+ column +'" data-index="'+ index +'"></div>'
	// }
	for(let y=1; y<=inputSizeWidthValue; y++) {
		chess[y]=[]
		chessTable+='<div class="row">'
		for(let x=1; x<=inputSizeHeightValue; x++) {
			let index = ((y-1) * inputSizeWidthValue) + x
			chess[y][x]=true
			chessTable+='<div class="column" data-row="'+ y +'" data-column="'+ x +'" data-index="'+ index +'"></div>'
		}
		chessTable+='</div>'
	}
	boards.forEach(function(board) {
		board.innerHTML=chessTable
	})
	// alert(inputSizeWidthValue + "/" + inputSizeHeightValue)
	sectionForm.style.display="none"
	sectionRook.style.display="none"
	sectionBoard.style.display="block"
	boards.forEach(function(board) {
		let boardItems = board.querySelectorAll(".column")
		boardItems.forEach(function(column) {
			column.addEventListener("click", function() {
				let row=this.getAttribute('data-row')
				let column=this.getAttribute('data-column')
				let index=this.getAttribute('data-index')
				let currentSection = this.parentElement.parentElement.parentElement
				if(currentSection.classList.contains("section-board")) {
					boards.forEach(function(_board) {
						console.log(_board)
						let item=_board.querySelector("[data-index='"+ index +"']")
						if(item) {
							item.classList.toggle("disable")
						}
					})
					if(chess[row][column] === true) {
						chess[row][column]=false
					}
					else {
						chess[row][column]=true
					}
				}
				else if(currentSection.classList.contains("section-rook")) {
					if(!this.classList.contains("disable")) {
						boardItems.forEach(function(_column) {
							_column.classList.remove("rook")
						})
						this.classList.add("rook")
					}
				}
			})
		})
	})
})
let boardButton = sectionBoard.querySelector(".board-button")
boardButton.addEventListener("click", function(){
	sectionForm.style.display="none"
	sectionBoard.style.display="none"
	sectionRook.style.display="block"
	// let boardItem = boards[1].querySelector(".column")
	// boardItem.classList.add("rook")
})
