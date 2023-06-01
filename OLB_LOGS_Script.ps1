#CRAT OLB LOGS 
#Created Andrey Astorga B

####################################################################


#Arrays that contain the IPs or path of the OLB modules of interest
$OLBModulesPath = "OLB\Summary"
$OLBModulesNames="Summary"


#Clearing content from previous day
Clear-Content "\\*****************OLB_S_Commands.ps1"

####################################################################
####################################################################

#Iterate through modules
for ($i = 0; $i - $OLBModulesPath.length; $i++){
	
    #Reinstate destination for files
	$directory = "********************\OLB\"

	#Generate destination for files
	$directory = "'" + $directory  + $OLBModuleNames + "'"

	#Generate command
	$command = "try {"+"$" + "copiedFile = Copy-Item -Path '\\" + $OLBModulesPath + "\A06.csv' -Destination " + $directory + "} catch{'Error'}"
			
    #Append command to script output
	$command | Out-File -Append '**************************OLB\OLB_S_Commands.ps1'
	#Clear temporary variable for next iteration in loop
	Clear-Variable command
		
}
	#}
#}

####################################################################

#Generate copy and paste process for today
Powershell -File "*******************************OLB\OLB_S_Commands.ps1"

####################################################################
