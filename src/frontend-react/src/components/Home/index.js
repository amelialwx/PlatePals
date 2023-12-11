import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import Button from '@material-ui/core/Button';
import Icon from '@material-ui/core/Icon';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import TextField from '@material-ui/core/TextField';



import DataService from "../../services/DataService";
import styles from './styles';
import TOC from '../TOC';


const Home = (props) => {
    const { classes } = props;

    console.log("================================== Home ======================================");


    // Component States
    const [contentList, setContentList] = useState([]);

    // Setup Component
    useEffect(() => {

    }, []);

    // Handlers



    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth={false} className={classes.container}>
                    <Grid container spacing={3}>
                        <Grid item xs={4}>
                            <TOC />
                        </Grid>
                        <Grid item xs={8}>
                        <h1>&#129302; Hello! I am your PlatePals AI Assistant!</h1>
                        <h2>The goal of PlatePals is to develop a machine learning application that accurately identifies the types of food present in a user-uploaded image. Based on the foods identified, the application will provide the user with the predicted food label and can offer relevant nutritional information and personalized dietary recommendations via a ChatBot using state-of-the-art GPT models.</h2>
                        <h2>Simply upload an image on the Upload Image tab to get started!</h2>
                        </Grid>
                    </Grid>
                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(Home);