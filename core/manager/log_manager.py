# -*- encoding: utf-8 -*-
import datetime
import os
from pathlib import Path
import datetime
from typing import Any , Optional

import wandb
from rich.panel import Panel
from rich.console import Console
from rich.pretty import Pretty
from rich.rule import Rule
from loguru import logger

from .argument_manager import ArgumentManager


class LogManager :
    """
    A rich-based logging and printing manager that provides enhanced console output
    and logging capabilities.

    This class manages console output using rich library and integrates with loguru
    for logging functionality. It supports both console output and file logging.
    """

    _console: Optional [ Console ] = None
    _log_file: Optional [ Path ] = None
    WANDB: bool = False
    TENSORBOARD: bool = False

    @classmethod
    def init ( cls , output_path: str = None ) -> None :
        """
        Initialize the PrintManager with console and logging configuration.

        Args:
            output_path: The path where log files should be stored. If None, will use ArgumentManger.get_output_path()

        This method should be called before using any other methods of the class.
        """

        try :
            if output_path is None :
                path = ArgumentManager.get ( "work_dir" )
            if not os.path.exists ( path ) :
                os.makedirs ( path )

            cls._console = Console (
                color_system = 'auto' ,
                log_time_format = "[%Y-%m-%d %H:%M:%S]"
            )
            log_path = Path ( path ) / f"log_{datetime.datetime.now ( ).strftime ( '%Y%m%d_%H%M%S' )}.log"
            cls._log_file = log_path
            logger.add ( str ( log_path ) )
        except Exception as e :
            logger.error ( f"Failed to initialize PrintManager: {str ( e )}" )
            raise

        try :
            wandb_config = ArgumentManager.get ( "wandb" )
            if isinstance ( wandb_config , dict ) and wandb_config.get ( "enable" , False ) :
                cls.init_wandb ( )
        except Exception as e :
            cls.WANDB = False
            logger.warning ( f"Failed to initialize wandb, continuing without it: {str ( e )}" )

        LogManager.info_manager ( )

    @classmethod
    def init_wandb ( cls ) -> None :
        """
        Initialize Weights & Biases (wandb) for experiment tracking.

        This method sets up wandb with the project name and configuration.
        """
        try :
            output_path = ArgumentManager.get ( "work_dir" )
            if not os.path.exists ( os.path.join ( output_path , "wandb" ) ) :
                os.makedirs ( os.path.join ( output_path , "wandb" ) )
            wandb.init (
                project = ArgumentManager.get ( "wandb" ).get ( "project" , "default_project" ) ,
                config = ArgumentManager.get ( ) ,
                entity = ArgumentManager.get ( "wandb" ).get ( "entity" , "default_entity" ) ,
                dir = os.path.join ( output_path , "wandb/" ) ,
            )
            cls.WANDB = True
        except Exception as e :
            cls.WANDB = False
            logger.warning ( f"Failed to initialize wandb, continuing without it: {str ( e )}" )

    @classmethod
    def info_manager ( cls ) -> None :
        LogManager.info_panel ( vars( ArgumentManager.get ( ) ) , "Argument" )
        # LogManager.info_panel ( ConfigManager.get ( ) , "Config" )

    @classmethod
    def info_title ( cls , data: str , characters: str = "*" ) -> None :
        """
        Print a title with a rule to both console and log file.

        Args:
            data: The title text to display
            characters: The character to use for the rule (default: "*")
        """
        try :
            rule = Rule ( data , characters = characters )
            if cls._log_file :
                with cls._log_file.open ( "a" ) as report_file :
                    console = Console (
                        color_system = 'auto' ,
                        file = report_file ,
                        log_time_format = "[%Y-%m-%d %H:%M:%S]"
                    )
                    console.print ( rule )
            if cls._console :
                cls._console.print ( rule )
        except Exception as e :
            logger.error ( f"Failed to print info title: {str ( e )}" )

    @classmethod
    def info_panel ( cls , data: Any , title: str = "None" ) -> None :
        """
        Print data in a panel format to both console and log file.

        Args:
            data: The data to display in the panel
            title: The title of the panel (default: "None")
        """
        try :
            pretty = Pretty ( data )
            if cls._log_file :
                with cls._log_file.open ( "a" ) as report_file :
                    console = Console (
                        color_system = 'auto' ,
                        file = report_file ,
                        log_time_format = "[%Y-%m-%d %H:%M:%S]"
                    )
                    console.print ( Panel ( pretty , title = title ) )
            if cls._console :
                cls._console.print ( Panel ( pretty , title = title ) )
        except Exception as e :
            logger.error ( f"Failed to print info panel: {str ( e )}" )

    @classmethod
    def info ( cls , data: Any ) -> None :
        """
        Log an info message.

        Args:
            data: The message to log
        """
        logger.info ( data )
        if cls.WANDB and type ( data ) is dict :
            wandb.log ( data )

    @classmethod
    def warning ( cls , data: Any ) -> None :
        """
        Log a warning message.

        Args:
            data: The warning message to log
        """
        logger.warning ( data )
        if cls.WANDB and type ( data ) is dict :
            wandb.warning ( data )

    @classmethod
    def error ( cls , data: Any ) -> None :
        """
        Log an error message.

        Args:
            data: The error message to log
        """
        logger.error ( data )
        if cls.WANDB and type ( data ) is dict :
            wandb.error ( data )
